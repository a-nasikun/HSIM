# Spectra - header-only sparse eigensolvers, used by the tests as an independent reference
if(TARGET Spectra::Spectra)
	return()
endif()

include(FetchContent)
FetchContent_Declare(
	spectra
	GIT_REPOSITORY https://github.com/yixuan/spectra.git
	GIT_TAG        v1.0.1
	GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(spectra)
if(NOT spectra_POPULATED)
	FetchContent_Populate(spectra)
endif()
add_library(hsim_spectra INTERFACE)
add_library(Spectra::Spectra ALIAS hsim_spectra)
target_include_directories(hsim_spectra SYSTEM INTERFACE ${spectra_SOURCE_DIR}/include)
