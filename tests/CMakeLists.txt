# Test-suite for the HSIM library (CPU build).
cmake_minimum_required(VERSION 3.16)
project(HSIM_Tests CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../HSIM/cmake)

# The current library headers still depend on the libigl viewer.
option(LIBIGL_WITH_OPENGL      "Use OpenGL" ON)
option(LIBIGL_WITH_OPENGL_GLFW "Use GLFW"   ON)
option(IF_VERBOSE "Print the intermediate steps" OFF)
# libigl v2.3.0 headers miss <cstdint> with recent GCC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include cstdint")

include(libigl)
add_subdirectory(../HSIM HSIM-build)
target_link_libraries(HSIM PUBLIC igl::opengl_glfw)

include(FetchContent)
FetchContent_Declare(doctest GIT_REPOSITORY https://github.com/doctest/doctest.git GIT_TAG v2.4.11)
FetchContent_MakeAvailable(doctest)

add_executable(hsim_tests test_main.cpp test_regression.cpp)
target_include_directories(hsim_tests PRIVATE ../HSIM)
target_compile_definitions(hsim_tests PRIVATE HSIM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(hsim_tests PRIVATE HSIM doctest::doctest)

enable_testing()
add_test(NAME hsim_tests COMMAND hsim_tests)
