cmake_minimum_required(VERSION 3.5)

project(simplest)

set(CMAKE_CXX_STANDARD 20)

find_package(SDL2 REQUIRED)
find_package(glm REQUIRED)

add_executable(simplest 
    src/main.cpp
    src/gl/glad.c

    src/imgui/imgui.cpp
    src/imgui/imgui_demo.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_impl_sdl2.cpp
    src/imgui/imgui_impl_opengl3.cpp
    src/imgui/imgui_tables.cpp
)

target_link_libraries(simplest 
    PRIVATE SDL2::SDL2 
    PRIVATE glm::glm 
)

target_include_directories(simplest 
    PRIVATE src
)

target_compile_definitions(simplest
    PRIVATE SDL_MAIN_HANDLED=1
)