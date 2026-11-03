# HSIM - Hierarchical Subspace Iteration Method for Laplace--Beltrami eigenproblems
cmake_minimum_required(VERSION 3.20)
project(HSIM
	VERSION 2.0.0
	DESCRIPTION "Hierarchical Subspace Iteration Method for Laplace--Beltrami eigenproblems"
	HOMEPAGE_URL "https://github.com/a-nasikun/HSIM"
	LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
	set(HSIM_TOPLEVEL_PROJECT ON)
else()
	set(HSIM_TOPLEVEL_PROJECT OFF)
endif()

# Options
# #################################################################################################
option(HSIM_WITH_CUDA      "Solve the dense (Rayleigh-Ritz) eigenproblems on the GPU with cuSOLVER" OFF)
option(HSIM_BUILD_EXAMPLES "Build the GUI example (requires OpenGL/GLFW)"                              ${HSIM_TOPLEVEL_PROJECT})
option(HSIM_BUILD_CLI      "Build the command-line program hsim_cli"                                 ${HSIM_TOPLEVEL_PROJECT})
option(HSIM_BUILD_TESTS    "Build the test-suite"                                                   ${HSIM_TOPLEVEL_PROJECT})
option(HSIM_BUILD_BENCHMARKS "Build the benchmark program"                                           OFF)
option(HSIM_BUILD_DOCS      "Add the target hsim_docs (Doxygen API documentation)"                    OFF)
option(HSIM_WARNINGS_AS_ERRORS "Treat compiler warnings in HSIM code as errors"                       OFF)

if(HSIM_TOPLEVEL_PROJECT AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "HSIM: no build type given, defaulting to Release")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Dependencies (fetched at configure time, pinned versions)
# #################################################################################################
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libigl)
include(HSIMWarnings)

# Targets
# #################################################################################################
add_subdirectory(HSIM)

if(HSIM_BUILD_EXAMPLES)
	add_subdirectory(Example/1_HSIM_GUI)
endif()

if(HSIM_BUILD_CLI)
	add_subdirectory(Example/2_HSIM_CLI)
endif()

if(HSIM_BUILD_BENCHMARKS)
	add_subdirectory(benchmarks)
endif()

if(HSIM_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()

if(HSIM_BUILD_DOCS)
	find_package(Doxygen REQUIRED)
	add_custom_target(hsim_docs
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/docs
		COMMAND ${CMAKE_COMMAND} -E env sh -c "( cat docs/Doxyfile; echo OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/docs ) | ${DOXYGEN_EXECUTABLE} -"
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		COMMENT "Generating the API documentation in ${CMAKE_BINARY_DIR}/docs"
		VERBATIM)
endif()
