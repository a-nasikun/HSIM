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

# Only the modules HSIM needs: the core library uses igl::core only; the viewer is needed by the GUI example.
set(LIBIGL_OPENGL ${HSIM_BUILD_EXAMPLES} CACHE BOOL "" FORCE)
set(LIBIGL_GLFW   ${HSIM_BUILD_EXAMPLES} CACHE BOOL "" FORCE)
set(LIBIGL_IMGUI  ${HSIM_BUILD_EXAMPLES} CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libigl)

# Treat the libigl headers as system headers: warnings in third-party code are not ours to fix.
foreach(target IN ITEMS igl_core igl_glfw igl_imgui igl_opengl)
	if(TARGET ${target})
		get_target_property(dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
		if(dirs)
			set_target_properties(${target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${dirs}")
		endif()
	endif()
endforeach()
