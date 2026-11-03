# HSIM core library
add_library(HSIM
	include/hsim/common.h
	include/hsim/dense_eigensolver.h
	include/hsim/hierarchy.h
	include/hsim/hsim.h
	include/hsim/log.h
	include/hsim/mesh.h
	include/hsim/options.h
	include/hsim/prolongation.h
	include/hsim/sampling.h
	include/hsim/solver.h
	include/hsim/subspace_iteration.h
	src/dense_eigensolver.cpp
	src/hierarchy.cpp
	src/logger.h
	src/mesh.cpp
	src/options.cpp
	src/prolongation.cpp
	src/sampling.cpp
	src/solver.cpp
	src/subspace_iteration.cpp
	src/timer.h
)
add_library(HSIM::HSIM ALIAS HSIM)

target_include_directories(HSIM
	PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(HSIM PUBLIC cxx_std_17)
target_link_libraries(HSIM PUBLIC igl::core)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(HSIM PRIVATE OpenMP::OpenMP_CXX)
else()
	message(STATUS "HSIM: OpenMP not found, the prolongation is built single-threaded")
endif()

hsim_set_warnings(HSIM)

if(HSIM_WITH_CUDA)
	find_package(CUDAToolkit REQUIRED)
	target_sources(HSIM PRIVATE src/dense_eigensolver_cuda.cpp)
	target_compile_definitions(HSIM PRIVATE HSIM_WITH_CUDA)
	target_link_libraries(HSIM PRIVATE CUDA::cudart CUDA::cusolver)
	message(STATUS "HSIM: dense eigenproblems are solved with cuSOLVER (CUDA ${CUDAToolkit_VERSION})")
else()
	message(STATUS "HSIM: dense eigenproblems are solved on the CPU (Eigen)")
endif()
