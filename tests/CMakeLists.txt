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
