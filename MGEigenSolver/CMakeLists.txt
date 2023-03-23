cmake_minimum_required (VERSION 3.10 FATAL_ERROR)
project(MultiLevelEigenSolver LANGUAGES CXX)

#set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
cmake_policy(SET CMP0076 NEW)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(libigl)
# OpenMP, OpenGL
# #################################################################################################
find_package(OpenMP)
find_package(OpenGL)

# Check for CUDA and set the flags
find_package(CUDA QUIET)
if(CUDA_FOUND)
  # Add the necessary libraries and include directories for CUDA
  include_directories(${CUDA_INCLUDE_DIRS})
  add_definitions(-DUSE_CUDA)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30" CACHE STRING "nvcc flags" FORCE)
  set(LIBS ${LIBS} ${CUDA_LIBRARIES})
endif()

					
# EIGEN
# #################################################################################################
#add_library(eigen INTERFACE)
#target_include_directories(eigen INTERFACE ../libigl/external/eigen)
#if(MSVC)
#	target_sources(eigen INTERFACE ../libigl/external/eigen/debug/msvc/eigen.natvis)
#endif()

set(SRC_FILES
	"MLSolverFunctions.cpp"
	"Basis.cpp"
	"EigenSolver.cpp"
	"Sampling.cpp"
	"Utility.cpp"
)

set(HEADER_FILES
	"MLSolverFunctions.h"
	"Basis.h"
	"EigenSolver.h"
	"Sampling.h"
	"Utility.h"
) 

add_library(MGEigenSolver ${SRC_FILES} ${HEADER_FILES})
include_directories("${PROJECT_SOURCE_DIR}/../libigl/external/glad/include" "${PROJECT_SOURCE_DIR}/../libigl/external/glfw/include")
include_directories("C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2019.1.144/windows/mkl/include/")
include_directories("C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2019.1.144/windows/mkl/include/intel64/")
include_directories("${PROJECT_SOURCE_DIR}/../../build/_deps/spectra-src/include")
add_compile_definitions(_USE_MATH_DEFINES)
target_include_directories(MGEigenSolver PUBLIC)
target_link_libraries(MGEigenSolver PUBLIC igl::core OpenMP::OpenMP_CXX)
#if(MSVC)
# 	target_compile_definitions(${PROJECT_NAME} PUBLIC _USE_MATH_DEFINES)
#endif()


# GLAD
# #################################################################################################
### add_subdirectory(../libigl/external/glad)

# GLFW
# #################################################################################################
### option(GLFW_BUILD_EXAMPLES "" OFF)
### option(GLFW_BUILD_TESTS    "" OFF)
### option(GLFW_BUILD_DOCS     "" OFF)
### option(GLFW_INSTALL        "" OFF)
### option(GLFW_VULKAN_STATIC  "" OFF)
### option(GLFW_USE_OSMESA     "" OFF)
### option(GLFW_USE_HYBRID_HPG "" OFF)
### option(GLFW_USE_WAYLAND    "" OFF)
### option(GLFW_USE_MIR        "" OFF)
### add_subdirectory(../libigl/external/glfw)

# IMGUI
# #################################################################################################
### add_subdirectory(../libigl/external/imgui)

# IGL
# #################################################################################################
### add_library(../libigl INTERFACE)
### target_include_directories(../libigl INTERFACE libigl/include)

# Multi-level eigen solver
# #################################################################################################
### add_executable(MultiLevelEigenSolver
### 	"Main.cpp"
### 	"MLEigenSolver.h"
### 	"MLEigenSolver.cpp"
### 	"MLSolverFunctions.h"
### 	"MLSolverFunctions.cpp"
### 	"MLEigSolverGUI.h"
### 	"Basis.h"
### 	"Basis.cpp"
### 	"EigenSolver.h"
### 	"EigenSolver.cpp"
### 	"Sampling.h"
### 	"Sampling.cpp"
### 	"Utility.h"
### 	"Utility.cpp"
### 	"Visualization.h"
### 	"Visualization.cpp"
### 	)
### target_link_libraries(MultiLevelEigenSolver
### 	PUBLIC libigl
### 	PUBLIC eigen
### 	PUBLIC glad
### 	PUBLIC glfw
### 	PUBLIC imgui
### 	PUBLIC tetgen
### 	PUBLIC OpenGL::GL
### 	PUBLIC OpenMP::OpenMP_CXX
### 	PUBLIC cudart
### 	PUBLIC cublas
### 	PUBLIC cusolver
### 	PUBLIC cusparse	
### 	)
### set_target_properties(MultiLevelEigenSolver PROPERTIES
### 	VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin
### 	RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_CURRENT_LIST_DIR}/bin>)
### set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MultiLevelEigenSolver)
### if(MSVC)
### 	target_compile_definitions(MultiLevelEigenSolver PUBLIC _USE_MATH_DEFINES)
### endif()
