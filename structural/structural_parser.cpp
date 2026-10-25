// structural_parser.cpp - Phase 1: the lossless tokenizer and the HST1 token image.
//
// The tokenizer is a total scanner: it never fails and never drops a byte.
// Each token is the longest match of its class at the current position, capped
// at kMaxTokenBytes:
//   STRUCTURE    "\n", "\r\n", ( ) [ ] { } , ;
//   INDENTATION  runs of spaces, tabs, \f, \v and lone \r
//   KEYWORD      identifier-shaped words in the language's keyword table
//   IDENTIFIER   [A-Za-z_\x80-\xFF][A-Za-z0-9_\x80-\xFF]*
//   LITERAL      numbers, '...' "..." `...` strings, // and /* */ comments
//   OPERATOR     the longest of the multi-byte operators below, else any single byte
#include "structural/structural_parser.hpp"

#include "structural/token_stream.hpp"

#include <algorithm>
#include <cctype>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace hce::structural {
namespace {

using hc::u32;
using hc::u64;

template <std::size_t N>
constexpr std::array<std::string_view, N> sorted(std::array<std::string_view, N> a) {
  std::sort(a.begin(), a.end());
  return a;
}

// The union of the tables below: what tokenize() without a language uses.
constexpr auto kGeneric = sorted(std::to_array<std::string_view>({
    "False", "None", "True", "abstract", "alignas", "alignof", "and", "as", "assert", "async",
    "auto", "await", "bool", "break", "case", "catch", "char", "class", "co_await", "co_return", "co_yield",
    "const", "const_cast", "consteval", "constexpr", "constinit", "continue", "crate", "decltype", "def",
    "default", "defer", "del", "delete", "do", "double", "dyn", "dynamic_cast", "elif", "else", "enum", "except",
    "explicit", "export", "extends", "extern", "false", "final", "finally", "float", "fn", "for", "friend", "from",
    "func", "function", "global", "go", "goto", "if", "impl", "implements", "import", "in", "inline", "instanceof",
    "int", "interface", "is", "lambda", "let", "long", "loop", "match", "mod", "module", "mut", "namespace", "new",
    "noexcept", "nonlocal", "not", "null", "nullptr", "operator", "or", "override", "package", "pass", "private",
    "protected", "pub", "public", "raise", "range", "register", "reinterpret_cast", "requires", "return", "self",
    "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "super", "switch", "template",
    "this", "throw", "throws", "trait", "true", "try", "type", "typedef", "typeid", "typename", "typeof",
    "undefined", "union", "unsafe", "unsigned", "use", "using", "var", "virtual", "void", "volatile", "where",
    "while", "with", "yield"}));

constexpr auto kC = sorted(std::to_array<std::string_view>({
    "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Noreturn", "_Static_assert",
    "_Thread_local", "alignas", "alignof", "auto", "bool", "break", "case", "char", "const", "constexpr",
    "continue", "default", "do", "double", "else", "enum", "extern", "false", "float", "for", "goto", "if",
    "inline", "int", "long", "nullptr", "register", "restrict", "return", "short", "signed", "sizeof", "static",
    "static_assert", "struct", "switch", "thread_local", "true", "typedef", "typeof", "union", "unsigned", "void",
    "volatile", "while"}));

constexpr auto kCpp = sorted(std::to_array<std::string_view>({
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
    "char", "char16_t", "char32_t", "char8_t", "class", "co_await", "co_return", "co_yield", "compl", "concept",
    "const", "const_cast", "consteval", "constexpr", "constinit", "continue", "decltype", "default", "delete", "do",
    "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "final", "float", "for",
    "friend", "goto", "if", "import", "inline", "int", "long", "module", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "override", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast",
    "struct", "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
    "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"}));

constexpr auto kPython = sorted(std::to_array<std::string_view>({
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "case", "class", "continue", "def",
    "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
    "match", "nonlocal", "not", "or", "pass", "raise", "return", "try", "type", "while", "with", "yield"}));

constexpr auto kJavaScript = sorted(std::to_array<std::string_view>({
    "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do",
    "else", "export", "extends", "false", "finally", "for", "function", "get", "if", "import", "in", "instanceof",
    "let", "new", "null", "of", "return", "set", "static", "super", "switch", "this", "throw", "true", "try",
    "typeof", "undefined", "var", "void", "while", "with", "yield"}));

constexpr auto kTypeScript = sorted(std::to_array<std::string_view>({
    "abstract", "any", "as", "asserts", "async", "await", "bigint", "boolean", "break", "case", "catch", "class",
    "const", "continue", "debugger", "declare", "default", "delete", "do", "else", "enum", "export", "extends",
    "false", "finally", "for", "function", "get", "if", "implements", "import", "in", "infer", "instanceof",
    "interface", "is", "keyof", "let", "module", "namespace", "never", "new", "null", "number", "object", "of",
    "private", "protected", "public", "readonly", "return", "satisfies", "set", "static", "string", "super",
    "switch", "symbol", "this", "throw", "true", "try", "type", "typeof", "undefined", "unique", "unknown", "var",
    "void", "while", "with", "yield"}));

constexpr auto kJava = sorted(std::to_array<std::string_view>({
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue",
    "default", "do", "double", "else", "enum", "extends", "false", "final", "finally", "float", "for", "goto", "if",
    "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "null", "package", "permits",
    "private", "protected", "public", "record", "return", "sealed", "short", "static", "strictfp", "super",
    "switch", "synchronized", "this", "throw", "throws", "transient", "true", "try", "var", "void", "volatile",
    "while", "yield"}));

constexpr auto kGo = sorted(std::to_array<std::string_view>({
    "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "false", "for", "func",
    "go", "goto", "if", "import", "interface", "iota", "map", "nil", "package", "range", "return", "select",
    "struct", "switch", "true", "type", "var"}));

constexpr auto kRust = sorted(std::to_array<std::string_view>({
    "Self", "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false",
    "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "self",
    "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where", "while"}));

using Keywords = std::span<const std::string_view>;

Keywords keywords_for(Language lang) noexcept {
  switch (lang) {
    case Language::c: return kC;
    case Language::cpp: return kCpp;
    case Language::python: return kPython;
    case Language::javascript: return kJavaScript;
    case Language::typescript: return kTypeScript;
    case Language::java: return kJava;
    case Language::go: return kGo;
    case Language::rust: return kRust;
    case Language::generic: break;
  }
  return kGeneric;
}

constexpr std::array<std::string_view, 9> kOps3{"<<=", ">>=", "...", "->*", "<=>", "**=", ">>>", "===", "!=="};
constexpr std::array<std::string_view, 28> kOps2{"::", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=",
                                                 "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
                                                 "=>", "**", ".*", "##", ":=", "?.", "??", "<-"};

constexpr bool is_ident_start(unsigned c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c >= 0x80;
}
constexpr bool is_digit(unsigned c) noexcept { return c >= '0' && c <= '9'; }
constexpr bool is_ident(unsigned c) noexcept { return is_ident_start(c) || is_digit(c); }
constexpr bool is_hspace(unsigned c) noexcept { return c == ' ' || c == '\t' || c == '\f' || c == '\v' || c == '\r'; }
constexpr bool is_structure(unsigned c) noexcept {
  return c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == ',' || c == ';';
}

bool is_keyword(Keywords kw, std::string_view w) noexcept { return std::binary_search(kw.begin(), kw.end(), w); }

// One token starting at i (i < s.size()); advances i by 1..kMaxTokenBytes.
TokenType scan(std::string_view s, std::size_t& i, Keywords kw) noexcept {
  const std::size_t n = s.size(), start = i;
  const std::size_t cap = std::min(n, start + kMaxTokenBytes);
  const auto at = [s](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned c = at(i);

  if (c == '\n') {
    ++i;
    return TokenType::STRUCTURE;
  }
  if (c == '\r' && i + 1 < n && at(i + 1) == '\n') {
    i += 2;
    return TokenType::STRUCTURE;
  }
  if (is_hspace(c)) {
    while (i < cap && is_hspace(at(i)) && !(at(i) == '\r' && i + 1 < n && at(i + 1) == '\n')) ++i;
    return TokenType::INDENTATION;
  }
  if (is_ident_start(c)) {
    while (i < cap && is_ident(at(i))) ++i;
    return is_keyword(kw, s.substr(start, i - start)) ? TokenType::KEYWORD : TokenType::IDENTIFIER;
  }
  if (is_digit(c) || (c == '.' && i + 1 < n && is_digit(at(i + 1)))) {
    ++i;
    while (i < cap) {
      const unsigned d = at(i), p = at(i - 1);
      if (is_ident(d) || d == '.' || ((d == '+' || d == '-') && (p == 'e' || p == 'E' || p == 'p' || p == 'P')))
        ++i;
      else
        break;
    }
    return TokenType::LITERAL;
  }
  if (c == '"' || c == '\'' || c == '`') {
    ++i;
    while (i < cap) {
      const unsigned d = at(i);
      if (d == '\\') {
        i = std::min(i + 2, cap);
        continue;
      }
      if (d == '\n' && c != '`') break;  // unterminated: the newline stays a token of its own
      ++i;
      if (d == c) break;
    }
    return TokenType::LITERAL;
  }
  if (c == '/' && i + 1 < n && at(i + 1) == '/') {
    i += 2;
    while (i < cap && at(i) != '\n') ++i;
    return TokenType::LITERAL;
  }
  if (c == '/' && i + 1 < n && at(i + 1) == '*') {
    i += 2;
    while (i < cap) {
      if (at(i) == '*' && i + 1 < cap && at(i + 1) == '/') {
        i += 2;
        break;
      }
      ++i;
    }
    return TokenType::LITERAL;
  }
  if (is_structure(c)) {
    ++i;
    return TokenType::STRUCTURE;
  }
  for (const std::string_view op : kOps3)
    if (s.substr(i, 3) == op) {
      i += 3;
      return TokenType::OPERATOR;
    }
  for (const std::string_view op : kOps2)
    if (s.substr(i, 2) == op) {
      i += 2;
      return TokenType::OPERATOR;
    }
  ++i;  // any other byte, printable or not
  return TokenType::OPERATOR;
}

constexpr u64 zigzag(std::int64_t v) noexcept {
  return (static_cast<u64>(v) << 1) ^ static_cast<u64>(v >> 63);
}
constexpr std::int64_t unzigzag(u64 v) noexcept {
  return static_cast<std::int64_t>(v >> 1) ^ -static_cast<std::int64_t>(v & 1);
}

}  // namespace

// ------------------------------------------------------------- Phase 1 API ---

std::vector<StructuralToken> StructuralPreprocessor::tokenize(const std::string& source_code) {
  return structural::tokenize(source_code, Language::generic);
}

std::vector<uint8_t> StructuralPreprocessor::serialize_tokens(const std::vector<StructuralToken>& tokens) {
  std::size_t bytes = kTokenStreamMagic.size() + 10;
  for (const StructuralToken& t : tokens) bytes += t.value.size() + 8;
  std::vector<std::uint8_t> out;
  out.reserve(bytes);
  for (const std::byte b : kTokenStreamMagic) out.push_back(static_cast<std::uint8_t>(b));
  put_varint(out, tokens.size());
  u32 prev_line = 1;
  for (const StructuralToken& t : tokens) {
    out.push_back(static_cast<std::uint8_t>(t.type));
    put_varint(out, t.value.size());
    out.insert(out.end(), t.value.begin(), t.value.end());
    put_varint(out, zigzag(std::int64_t{t.line_number} - std::int64_t{prev_line}));
    prev_line = t.line_number;
    put_varint(out, t.checked_integer_id);
  }
  return out;
}

// ------------------------------------------------------- Phase 2 additions ---

std::vector<StructuralToken> tokenize(std::string_view s, Language lang) {
  if (s.size() > kMaxSourceBytes) throw std::length_error("hce::structural::tokenize: input larger than kMaxSourceBytes");
  const Keywords kw = keywords_for(lang);
  std::vector<StructuralToken> out;
  out.reserve(s.size() / 4 + 1);
  std::unordered_map<std::string, u32> ids;  // key: type byte, then the value
  std::string key;
  u32 line = 1;
  for (std::size_t i = 0; i < s.size();) {
    const std::size_t start = i;
    const TokenType type = scan(s, i, kw);
    const std::string_view value = s.substr(start, i - start);
    key.assign(1, static_cast<char>(type));
    key.append(value);
    // At most one entry per token and kMaxSourceBytes < 2^32 tokens: the IDs fit.
    const auto [it, fresh] = ids.try_emplace(key, static_cast<u32>(ids.size()));
    out.push_back(StructuralToken{type, std::string(value), line, it->second});
    line += static_cast<u32>(std::count(value.begin(), value.end(), '\n'));
  }
  return out;
}

const char* to_string(Language lang) noexcept {
  switch (lang) {
    case Language::generic: return "generic";
    case Language::c: return "c";
    case Language::cpp: return "cpp";
    case Language::python: return "python";
    case Language::javascript: return "javascript";
    case Language::typescript: return "typescript";
    case Language::java: return "java";
    case Language::go: return "go";
    case Language::rust: return "rust";
  }
  return "unknown";
}

std::optional<Language> parse_language(std::string_view n) noexcept {
  if (n == "generic") return Language::generic;
  if (n == "c") return Language::c;
  if (n == "cpp" || n == "c++" || n == "cxx") return Language::cpp;
  if (n == "python" || n == "py") return Language::python;
  if (n == "javascript" || n == "js") return Language::javascript;
  if (n == "typescript" || n == "ts") return Language::typescript;
  if (n == "java") return Language::java;
  if (n == "go") return Language::go;
  if (n == "rust" || n == "rs") return Language::rust;
  return std::nullopt;
}

Language language_for_path(const std::filesystem::path& path) noexcept {
  try {
    std::string ext = path.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".c") return Language::c;
    if (ext == ".h" || ext == ".hh" || ext == ".hpp" || ext == ".hxx" || ext == ".cc" || ext == ".cpp" ||
        ext == ".cxx" || ext == ".ipp" || ext == ".inl" || ext == ".tcc")
      return Language::cpp;
    if (ext == ".py" || ext == ".pyi") return Language::python;
    if (ext == ".js" || ext == ".mjs" || ext == ".cjs" || ext == ".jsx") return Language::javascript;
    if (ext == ".ts" || ext == ".tsx" || ext == ".mts" || ext == ".cts") return Language::typescript;
    if (ext == ".java") return Language::java;
    if (ext == ".go") return Language::go;
    if (ext == ".rs") return Language::rust;
  } catch (...) {  // path conversion: fall back to the generic table
  }
  return Language::generic;
}

void put_varint(std::vector<std::uint8_t>& out, u64 v) {
  while (v >= 0x80) {
    out.push_back(static_cast<std::uint8_t>(v | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(v));
}

bool get_varint(hc::Bytes in, std::size_t& at, u64& v) noexcept {
  v = 0;
  for (int shift = 0; shift <= 63; shift += 7) {
    if (at >= in.size()) return false;
    const u64 b = static_cast<u64>(in[at++]);
    if (shift == 63 && b > 1) return false;  // more than 64 bits
    v |= (b & 0x7F) << shift;
    if ((b & 0x80) == 0) return true;
  }
  return false;
}

hc::Errc deserialize_tokens(hc::Bytes in, std::vector<StructuralToken>& out, const TokenStreamLimits& lim) noexcept {
  out.clear();
  const auto fail = [&out](hc::Errc e) {
    out.clear();
    return e;
  };
  try {
    if (in.size() < kTokenStreamMagic.size() || !std::equal(kTokenStreamMagic.begin(), kTokenStreamMagic.end(), in.begin()))
      return hc::Errc::bad_magic;
    std::size_t at = kTokenStreamMagic.size();
    u64 count = 0;
    if (!get_varint(in, at, count)) return hc::Errc::header_corrupt;
    if (count > lim.max_tokens) return hc::Errc::limit_output;
    if (count > (in.size() - at) / 4) return hc::Errc::header_corrupt;  // a token takes >= 4 bytes
    out.reserve(static_cast<std::size_t>(count));
    u64 total = 0;
    u32 line = 1;
    for (u64 k = 0; k < count; ++k) {
      if (at >= in.size()) return fail(hc::Errc::header_corrupt);
      const auto type = static_cast<std::uint8_t>(in[at++]);
      if (type >= kTokenTypes) return fail(hc::Errc::header_corrupt);
      u64 len = 0, zz = 0, id = 0;
      if (!get_varint(in, at, len)) return fail(hc::Errc::header_corrupt);
      if (len > lim.max_token_bytes) return fail(hc::Errc::limit_output);
      if (len > in.size() - at) return fail(hc::Errc::header_corrupt);
      total += len;
      if (total > lim.max_value_bytes) return fail(hc::Errc::limit_output);
      std::string value(reinterpret_cast<const char*>(in.data() + at), static_cast<std::size_t>(len));
      at += static_cast<std::size_t>(len);
      if (!get_varint(in, at, zz) || !get_varint(in, at, id)) return fail(hc::Errc::header_corrupt);
      const std::int64_t d = unzigzag(zz);
      if (d > 0xFFFFFFFFll || d < -0xFFFFFFFFll) return fail(hc::Errc::header_corrupt);
      const std::int64_t next = std::int64_t{line} + d;
      if (next < 0 || next > 0xFFFFFFFFll || id > 0xFFFFFFFFull) return fail(hc::Errc::header_corrupt);
      line = static_cast<u32>(next);
      out.push_back(StructuralToken{static_cast<TokenType>(type), std::move(value), line, static_cast<u32>(id)});
    }
    if (at != in.size()) return fail(hc::Errc::header_corrupt);  // bytes after the last token
    return hc::Errc::ok;
  } catch (const std::bad_alloc&) {
    return fail(hc::Errc::out_of_memory);
  }
}

std::string detokenize(const std::vector<StructuralToken>& tokens) {
  std::size_t n = 0;
  for (const StructuralToken& t : tokens) n += t.value.size();
  std::string s;
  s.reserve(n);
  for (const StructuralToken& t : tokens) s += t.value;
  return s;
}

void StructuralTracker::advance(TokenType type, std::string_view v) noexcept {
  constexpr u32 kMaxNesting = 1u << 20;
  StructuralContextState& s = state;
  if (type == TokenType::STRUCTURE) {
    if (v.find('\n') != std::string_view::npos) {
      at_line_start = true;
      s.indentation_level = 0;
      return;
    }
    if (v == "{") {
      if (s.scope_depth < kMaxNesting) ++s.scope_depth;
    } else if (v == "}") {
      if (s.scope_depth > 0) --s.scope_depth;
    } else if (v == "(" || v == "[") {
      if (s.block_brackets_count < kMaxNesting) ++s.block_brackets_count;
    } else if (v == ")" || v == "]") {
      if (s.block_brackets_count > 0) --s.block_brackets_count;
    }
  } else if (type == TokenType::INDENTATION && at_line_start) {
    s.indentation_level = static_cast<u32>(std::min<std::size_t>(v.size(), kMaxTokenBytes));
    return;
  }
  at_line_start = false;
}

}  // namespace hce::structural
