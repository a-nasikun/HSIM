# libigl (header-only use) and, through it, Eigen 3.4
if(TARGET igl::core)
	return()
endif()

include(FetchContent)
FetchContent_Declare(
	libigl
	GIT_REPOSITORY https://github.com/libigl/libigl.git
	GIT_TAG        v2.5.0
	GIT_SHALLOW    TRUE
)

# Only the modules HSIM needs. The viewer (GLFW/OpenGL) is currently still required by the core library.
set(LIBIGL_OPENGL ON  CACHE BOOL "" FORCE)
set(LIBIGL_GLFW   ON  CACHE BOOL "" FORCE)
set(LIBIGL_IMGUI  ${HSIM_BUILD_EXAMPLES} CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libigl)
