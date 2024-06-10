add_subdirectory(utils)
add_executable(orb_test orb_test.cpp)
target_link_libraries(orb_test ${OpenCV_LIBS} utils ${Sophus_LIBRARIES})
