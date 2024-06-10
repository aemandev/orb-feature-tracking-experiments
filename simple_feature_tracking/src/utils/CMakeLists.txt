# Create a library for utils
add_library(utils ORB_helper_functions.cpp)

# Specify include directories
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR}/include/utils)
