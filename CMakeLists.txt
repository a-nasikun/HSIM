cmake_minimum_required (VERSION 3.10 FATAL_ERROR)
project(MultiLevelEigenSolver LANGUAGES CXX CUDA)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
cmake_policy(SET CMP0076 NEW)

# OpenMP, OpenGL
# #################################################################################################
find_package(OpenMP)
find_package(OpenGL)

# Matlab
set (MATLAB_LIBRARY_DIR "" CACHE PATH "Matlab static library dirs (e.g. R2018a/extern/lib/win64/microsoft)")
set (MATLAB_INCLUDE_DIR "" CACHE PATH "Matlab include dir ((e.g. R2018a/extern/include)")
set (SPECTRA_INCLUDE_DIR "" CACHE PATH "Include library for the Spectra eigensolver")
include_directories(${MATLAB_INCLUDE_DIR})
include_directories(${SPECTRA_INCLUDE_DIR}/include)
link_directories(${MATLAB_LIBRARY_DIR})
					
# EIGEN
# #################################################################################################
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE libigl/external/eigen)
if(MSVC)
	target_sources(eigen INTERFACE libigl/external/eigen/debug/msvc/eigen.natvis)
endif()

# GLAD
# #################################################################################################
add_subdirectory(libigl/external/glad)

# GLFW
# #################################################################################################
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS    "" OFF)
option(GLFW_BUILD_DOCS     "" OFF)
option(GLFW_INSTALL        "" OFF)
option(GLFW_VULKAN_STATIC  "" OFF)
option(GLFW_USE_OSMESA     "" OFF)
option(GLFW_USE_HYBRID_HPG "" OFF)
option(GLFW_USE_WAYLAND    "" OFF)
option(GLFW_USE_MIR        "" OFF)
add_subdirectory(libigl/external/glfw)

# IMGUI
# #################################################################################################
add_subdirectory(libigl/external/imgui)

# IGL
# #################################################################################################
add_library(libigl INTERFACE)
target_include_directories(libigl INTERFACE libigl/include)

# TETGEN
# #################################################################################################
add_subdirectory(libigl/external/tetgen)

# Multi-level eigen solver
# #################################################################################################
add_executable(MultiLevelEigenSolver
	"MLSolver/Main.cpp"
	"MLSolver/MLEigSolver.h"
	"MLSolver/MLEigSolver.cpp"
	"MLSolver/MLEigSolverGUI.h"
	"MLSolver/Basis.h"
	"MLSolver/Basis.cpp"
	"MLSolver/EigenSolver.h"
	"MLSolver/EigenSolver.cpp"
	"MLSolver/Sampling.h"
	"MLSolver/Sampling.cpp"
	"MLSolver/Utility.h"
	"MLSolver/Utility.cpp"
	"MLSolver/Visualization.h"
	"MLSolver/Visualization.cpp"
	"MLSolver/Hessian.h"
	)
target_link_libraries(MultiLevelEigenSolver
	PUBLIC libigl
	PUBLIC eigen
	PUBLIC glad
	PUBLIC glfw
	PUBLIC imgui
	PUBLIC tetgen
	PUBLIC OpenGL::GL
	PUBLIC OpenMP::OpenMP_CXX
	PUBLIC cudart
	PUBLIC cublas
	PUBLIC cusolver
	PUBLIC cusparse
	libMatlabEngine.lib			# for Matlab stuff
	libMatlabDataArray.lib      # 
	libeng.lib                  # 
	libut.lib                   # 
	libmex.lib                  # 
	libmx.lib                   # for matlab stuff
	)
set_target_properties(MultiLevelEigenSolver PROPERTIES
	VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin
	RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_LIST_DIR}/bin>)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MultiLevelEigenSolver)
if(MSVC)
	target_compile_definitions(MultiLevelEigenSolver PUBLIC _USE_MATH_DEFINES)
endif()
