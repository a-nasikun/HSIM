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
option(HSIM_VERBOSE        "Print the intermediate steps of the solver"                             OFF)
option(HSIM_BUILD_EXAMPLES "Build the example applications (requires OpenGL/GLFW)"                  ${HSIM_TOPLEVEL_PROJECT})
option(HSIM_BUILD_TESTS    "Build the test-suite"                                                   ${HSIM_TOPLEVEL_PROJECT})

if(HSIM_TOPLEVEL_PROJECT AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "HSIM: no build type given, defaulting to Release")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Dependencies (fetched at configure time, pinned versions)
# #################################################################################################
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libigl)

# Targets
# #################################################################################################
add_subdirectory(HSIM)

if(HSIM_BUILD_EXAMPLES)
	add_subdirectory(Example/1_HSIM_GUI)
endif()

if(HSIM_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()
