# Test-suite for the HSIM library
include(doctest)
include(spectra)

add_executable(hsim_tests
	test_main.cpp
	test_support.h
	test_regression.cpp
	test_bugfixes.cpp
	test_mesh.cpp
	test_sampling.cpp
	test_prolongation.cpp
	test_dense.cpp
	test_solver.cpp
)
target_compile_definitions(hsim_tests PRIVATE HSIM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(hsim_tests PRIVATE HSIM::HSIM doctest::doctest Spectra::Spectra)
hsim_set_warnings(hsim_tests)

add_test(NAME hsim_tests COMMAND hsim_tests)

# Smoke tests of the command-line program
if(TARGET hsim_cli)
	add_test(NAME hsim_cli_help COMMAND hsim_cli --help)
	add_test(NAME hsim_cli_run
		COMMAND hsim_cli ${PROJECT_SOURCE_DIR}/Models/AIM894_Chinese_Dragon/894_Chinese_Dragon_2000.obj
		        --modes 10 --tol 1e-4 --max-iter 10 -v 0 -o ${CMAKE_CURRENT_BINARY_DIR}/cli_smoke)
	add_test(NAME hsim_cli_bad_option COMMAND hsim_cli --no-such-option)
	set_tests_properties(hsim_cli_bad_option PROPERTIES WILL_FAIL TRUE)
endif()

# Every public header must compile on its own (self-contained, no hidden include-order dependencies).
file(GLOB HSIM_PUBLIC_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/HSIM/include/hsim/*.h")
set(HEADER_CHECK_SOURCES "")
foreach(header IN LISTS HSIM_PUBLIC_HEADERS)
	get_filename_component(name ${header} NAME_WE)
	set(source "${CMAKE_CURRENT_BINARY_DIR}/header_check/${name}.cpp")
	file(CONFIGURE OUTPUT ${source} CONTENT "#include <hsim/${name}.h>\n#include <hsim/${name}.h>\n")
	list(APPEND HEADER_CHECK_SOURCES ${source})
endforeach()
add_library(hsim_header_check OBJECT ${HEADER_CHECK_SOURCES})
target_link_libraries(hsim_header_check PRIVATE HSIM::HSIM)
hsim_set_warnings(hsim_header_check)
