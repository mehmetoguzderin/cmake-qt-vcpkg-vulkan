#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ratio>

#include <QApplication>
#include <QCommandLineParser>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFileDialog>
#include <QLCDNumber>
#include <QLabel>
#include <QLibraryInfo>
#include <QLoggingCategory>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QTabWidget>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QVulkanFunctions>
#include <QVulkanInstance>
#include <QVulkanWindow>
#include <QWidget>

#include <qcontainerfwd.h>
#include <qdatetime.h>
#include <qlogging.h>
#include <qmatrix4x4.h>
#include <qsurfaceformat.h>
#include <qtmetamacros.h>
#include <qvectornd.h>
#include <qversionnumber.h>

#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_glsl.hpp>
#include <vulkan/vulkan_core.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

std::vector<uint32_t> compile_file(const std::string &source_name,
                                   shaderc_shader_kind kind) {
  std::string glsl_code;
  {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    options.AddMacroDefinition("MY_DEFINE", "1");
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    std::ifstream source_file(source_name);
    std::string source{std::istreambuf_iterator<char>(source_file),
                       std::istreambuf_iterator<char>()};

    shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(source, kind, source_name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
      std::cerr << module.GetErrorMessage();
      return std::vector<uint32_t>();
    }
    std::vector<uint32_t> spirv_code(module.cbegin(), module.cend());
    spirv_cross::CompilerGLSL glsl_compiler(spirv_code);

    // Set GLSL version and other necessary options
    spirv_cross::CompilerGLSL::Options glsl_options;
    glsl_options.version = 460;
    glsl_options.vulkan_semantics = true; // Enable Vulkan semantics
    glsl_compiler.set_common_options(glsl_options);

    // Convert and retrieve the GLSL code
    glsl_code = glsl_compiler.compile();
  }
  shaderc::Compiler final_compiler;
  shaderc::CompileOptions final_options;

  final_options.AddMacroDefinition("MY_DEFINE", "1");
  final_options.SetOptimizationLevel(shaderc_optimization_level_performance);

  shaderc::SpvCompilationResult final_module = final_compiler.CompileGlslToSpv(
      glsl_code, kind, source_name.c_str(), final_options);

  if (final_module.GetCompilationStatus() !=
      shaderc_compilation_status_success) {
    std::cerr << final_module.GetErrorMessage();
    return std::vector<uint32_t>();
  }
  return {final_module.cbegin(), final_module.cend()};
}
