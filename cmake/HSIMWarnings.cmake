# Compiler warnings for HSIM's own targets (third-party headers are included as system headers).
function(hsim_set_warnings target)
	if(MSVC)
		target_compile_options(${target} PRIVATE /W4 /permissive-)
		if(HSIM_WARNINGS_AS_ERRORS)
			target_compile_options(${target} PRIVATE /WX)
		endif()
	else()
		target_compile_options(${target} PRIVATE
			-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual
			-Wcast-align -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough)
		if(HSIM_WARNINGS_AS_ERRORS)
			target_compile_options(${target} PRIVATE -Werror)
		endif()
	endif()
endfunction()

# GCC reports false positives (maybe-uninitialized, array-bounds, ...) inside Eigen/libigl templates that are
# instantiated in our translation units (e.g. the header-only libigl viewer); silence those for such targets.
function(hsim_silence_third_party_flow_warnings target)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		target_compile_options(${target} PRIVATE
			-Wno-maybe-uninitialized -Wno-uninitialized -Wno-array-bounds -Wno-dangling-pointer)
	endif()
endfunction()
