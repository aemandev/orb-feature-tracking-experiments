find_package(Pangolin REQUIRED)
add_executable(optical_flow optical_flow.cpp OpticalFlowTracker.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS}
${Sophus_LIBRARIES}
g2o_core g2o_stuff)

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
target_link_libraries(direct_method Sophus::Sophus)
 
 