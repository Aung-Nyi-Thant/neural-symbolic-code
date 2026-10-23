add_library(hce_encoder STATIC planner.cpp)
target_link_libraries(hce_encoder PUBLIC hce_core PRIVATE hce_warnings)
