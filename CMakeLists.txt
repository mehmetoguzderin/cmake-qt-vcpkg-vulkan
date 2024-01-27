cmake_minimum_required(VERSION 3.20)
if(APPLE
    AND "${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin"
    AND "${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
    include_directories(/usr/local/include) # MoltenVK
    message(STATUS "Building for VK on macOS for macOS")
endif()
set(CMAKE_TOOLCHAIN_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(ProjectCxx)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(DCMTK CONFIG REQUIRED)
find_package(Qt6 CONFIG REQUIRED)
find_package(Qt6Gui CONFIG REQUIRED)
find_package(Qt6Network CONFIG REQUIRED)
find_package(Qt6OpenGL CONFIG REQUIRED)
find_package(Qt6OpenGLWidgets CONFIG REQUIRED)
find_package(Qt6Sql CONFIG REQUIRED)
find_package(Qt6Widgets CONFIG REQUIRED)
find_package(unofficial-shaderc CONFIG REQUIRED)
find_package(spirv_cross_core CONFIG REQUIRED)
find_package(spirv_cross_glsl CONFIG REQUIRED)
find_package(spirv_cross_hlsl CONFIG REQUIRED)
find_package(spirv_cross_msl CONFIG REQUIRED)
find_package(spirv_cross_reflect CONFIG REQUIRED)
find_package(spirv_cross_util CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(Taskflow CONFIG REQUIRED)
find_package(TIFF REQUIRED)
find_package(Vulkan REQUIRED)

qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# set(CMAKE_CXX_CLANG_TIDY clang-tidy)

add_library(LibCxx ${SOURCE_FILES})

qt_add_executable(MainCxx main.cpp)
set_target_properties(MainCxx PROPERTIES
    WIN32_EXECUTABLE TRUE
#     MACOSX_BUNDLE TRUE
)
target_link_libraries(MainCxx PRIVATE
    DCMTK::DCMTK
    Qt::Core Qt::Gui Qt::Network Qt::Widgets
    unofficial::shaderc::shaderc
    spirv-cross-core spirv-cross-glsl spirv-cross-msl
    spirv-cross-reflect spirv-cross-util
    Taskflow::Taskflow TIFF::TIFF
    Vulkan::Vulkan
)
