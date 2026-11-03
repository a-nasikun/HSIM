# Stage timings (not part of the test-suite)
add_executable(hsim_benchmark benchmark.cpp)
target_link_libraries(hsim_benchmark PRIVATE HSIM::HSIM)
hsim_set_warnings(hsim_benchmark)
