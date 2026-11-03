# Command-line example: compute eigenpairs and write them with a record of the run
add_executable(hsim_cli Main.cpp)
target_link_libraries(hsim_cli PRIVATE HSIM::HSIM)
hsim_set_warnings(hsim_cli)
