# HSIM core library
add_library(HSIM
	LBEigenSolver.cpp
	LBEigenSolver.h
	EigenSolver.cpp
	EigenSolver.h
	Sampling.cpp
	Sampling.h
	Utility.cpp
	Utility.h
)
add_library(HSIM::HSIM ALIAS HSIM)

target_include_directories(HSIM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(HSIM PUBLIC cxx_std_17)

find_package(OpenMP REQUIRED)
target_link_libraries(HSIM PUBLIC igl::core igl::glfw OpenMP::OpenMP_CXX)

if(MSVC)
	target_compile_definitions(HSIM PUBLIC _USE_MATH_DEFINES)
	target_compile_options(HSIM PRIVATE /W4)
else()
	target_compile_options(HSIM PRIVATE -Wall -Wextra)
endif()

if(HSIM_VERBOSE)
	target_compile_definitions(HSIM PRIVATE VERBOSE)
endif()

if(HSIM_WITH_CUDA)
	find_package(CUDAToolkit REQUIRED)
	target_compile_definitions(HSIM PRIVATE HSIM_WITH_CUDA)
	target_link_libraries(HSIM PRIVATE CUDA::cudart CUDA::cusolver CUDA::cublas)
	message(STATUS "HSIM: dense eigenproblems are solved with cuSOLVER (CUDA ${CUDAToolkit_VERSION})")
else()
	message(STATUS "HSIM: dense eigenproblems are solved on the CPU (Eigen)")
endif()
