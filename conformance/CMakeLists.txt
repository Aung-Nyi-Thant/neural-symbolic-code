# Deterministic input corpus, shared with the unit tests.
add_library(hce_corpus STATIC corpus.cpp)
target_include_directories(hce_corpus PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hce_corpus PRIVATE hce_warnings)

# Golden-vector harness: compresses the corpus with every profile and compares sizes, CRCs and
# probability-trace hashes against golden_v1.txt.
add_executable(hce_conformance hce_conformance.cpp)
target_link_libraries(hce_conformance PRIVATE hce_format hce_corpus hce_warnings)
add_test(NAME conformance COMMAND hce_conformance --check ${CMAKE_CURRENT_SOURCE_DIR}/golden_v1.txt)
