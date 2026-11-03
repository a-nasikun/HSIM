# doctest - single-header unit-test framework
if(TARGET doctest::doctest)
	return()
endif()

include(FetchContent)
FetchContent_Declare(
	doctest
	GIT_REPOSITORY https://github.com/doctest/doctest.git
	GIT_TAG        v2.4.11
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(doctest)
