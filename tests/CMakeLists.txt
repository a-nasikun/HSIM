# Test-suite for the HSIM library
include(doctest)

add_executable(hsim_tests
	test_main.cpp
	test_regression.cpp
	test_bugfixes.cpp
	test_support.h
)
target_compile_definitions(hsim_tests PRIVATE HSIM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(hsim_tests PRIVATE HSIM::HSIM doctest::doctest)

add_test(NAME hsim_tests COMMAND hsim_tests)
hsim_set_warnings(hsim_tests)

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
