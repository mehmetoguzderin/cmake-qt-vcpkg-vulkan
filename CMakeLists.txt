cmake_minimum_required(VERSION 3.27)
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

find_package(7zip CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(DCMTK CONFIG REQUIRED)
find_package(Qt6 CONFIG REQUIRED COMPONENTS
    Core Gui Network Widgets WebChannel WebSockets WebView
)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
# TODO: find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(unofficial-shaderc CONFIG REQUIRED)
find_package(spirv_cross_core CONFIG REQUIRED)
find_package(spirv_cross_glsl CONFIG REQUIRED)
find_package(spirv_cross_hlsl CONFIG REQUIRED)
find_package(spirv_cross_msl CONFIG REQUIRED)
find_package(spirv_cross_reflect CONFIG REQUIRED)
find_package(spirv_cross_util CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(Taskflow CONFIG REQUIRED)
# find_package(TIFF REQUIRED)
find_package(tinyfiledialogs CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
# zlib
find_package(ZLIB REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanobind)

# set(CMAKE_CXX_CLANG_TIDY clang-tidy)

add_library(LibCxx SHARED lib.cpp)
target_link_libraries(LibCxx PRIVATE
    Boost::boost ${Boost_LIBRARIES}
)

nanobind_add_module(PythonCxx STABLE_ABI NB_STATIC python.cpp)
target_link_libraries(PythonCxx PRIVATE LibCxx)

qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)

qt_add_executable(MainCxx main.cpp)
set_target_properties(MainCxx PROPERTIES
    WIN32_EXECUTABLE TRUE
#     MACOSX_BUNDLE TRUE
)
target_include_directories(MainCxx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/glm
    ${Boost_INCLUDE_DIRS}
)
target_link_libraries(MainCxx PRIVATE LibCxx)
target_link_libraries(MainCxx PRIVATE
    7zip::7zip ${Boost_LIBRARIES}
    DCMTK::DCMTK
    Qt::Core Qt::Gui Qt::Network Qt::Widgets Qt::WebChannel Qt::WebSockets Qt::WebView
    unofficial::shaderc::shaderc
    spirv-cross-core spirv-cross-glsl spirv-cross-msl
    spirv-cross-reflect spirv-cross-util
    Taskflow::Taskflow
    # TIFF::TIFF
    Vulkan::Vulkan
    tinyfiledialogs::tinyfiledialogs
    ZLIB::ZLIB
)

install(TARGETS MainCxx
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
qt_generate_deploy_app_script(
    TARGET MainCxx
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
