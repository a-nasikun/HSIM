cmake_minimum_required(VERSION 3.16)
project(HSIM_Example)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../MGEigenSolver/cmake)

# Libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          ON)
include(libigl)

include_directories("../build/_deps/libigl-src/external/imgui", "../build/_deps/libigl-src/external/imgui/backends", "../build/_deps/libigl-src/external/libigl-imgui")
include_directories("${PROJECT_SOURCE_DIR}/../MGEigenSolver")
include_directories("${PROJECT_SOURCE_DIR}/../libigl/external/imgui/imgui/examples/opengl3_example")
# Add MGSolver library
#add_subdirectory(../MGEigenSolver mgeigensolver-build)
add_subdirectory(../MGEigenSolver mgeigensolver-build)
file(GLOB SRC_FILES *.cpp *.h)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::core igl::opengl_glfw igl::opengl_glfw_imgui MGEigenSolver)
if(MSVC)
 	target_compile_definitions(${PROJECT_NAME} PUBLIC _USE_MATH_DEFINES)
endif()