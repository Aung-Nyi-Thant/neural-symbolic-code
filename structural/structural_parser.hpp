#ifndef STRUCTURAL_PARSER_HPP
#define STRUCTURAL_PARSER_HPP

#include <string>
#include <vector>
#include <cstdint>

namespace hce::structural {

enum class TokenType : uint8_t {
    KEYWORD = 0,
    IDENTIFIER = 1,
    LITERAL = 2,
    OPERATOR = 3,
    STRUCTURE = 4,
    INDENTATION = 5
};

struct StructuralToken {
    TokenType type;
    std::string value;
    uint32_t line_number;
    uint32_t checked_integer_id; // Checked integer ID for integer-only neural estimators
};

struct StructuralContextState {
    uint32_t scope_depth;
    uint32_t block_brackets_count;
    uint32_t indentation_level;
};

class StructuralPreprocessor {
public:
    StructuralPreprocessor() = default;
    ~StructuralPreprocessor() = default;

    std::vector<StructuralToken> tokenize(const std::string& source_code);
    std::vector<uint8_t> serialize_tokens(const std::vector<StructuralToken>& tokens);
};

} // namespace hce::structural

#endif // STRUCTURAL_PARSER_HPP
