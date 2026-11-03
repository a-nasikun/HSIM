# Interactive example: step-by-step hierarchy construction and visualization
add_executable(HSIM_GUI
	Main.cpp
	HSIMMenu.cpp
	HSIMMenu.h
	Visualization.cpp
	Visualization.h
)
target_link_libraries(HSIM_GUI PRIVATE HSIM::HSIM igl::glfw igl::imgui)
target_compile_definitions(HSIM_GUI PRIVATE HSIM_MODELS_DIR="${PROJECT_SOURCE_DIR}/Models")
hsim_set_warnings(HSIM_GUI)
hsim_silence_third_party_flow_warnings(HSIM_GUI)
