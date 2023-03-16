include(FetchContent)
FetchContent_Declare(
    spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
	GIT_TAG v1.0.0
)

# Note: In libigl v3.0.0, the following will become a one-liner:
# FetchContent_MakeAvailable(libigl)

FetchContent_GetProperties(spectra)
if(NOT spectra_POPULATED)
    FetchContent_Populate(spectra)
endif()
list(PREPEND CMAKE_MODULE_PATH "${spectra_SOURCE_DIR}/cmake")
include(${spectra_SOURCE_DIR}/include)