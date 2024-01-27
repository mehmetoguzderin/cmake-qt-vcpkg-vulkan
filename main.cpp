#include <fstream>
#include <iostream>

#include <QApplication>
#include <QCommandLineParser>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFileDialog>
#include <QLCDNumber>
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

#include <qtmetamacros.h>

#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_glsl.hpp>

class TriangleRenderer : public QVulkanWindowRenderer {
public:
  TriangleRenderer(QVulkanWindow *w, bool msaa = false);

  void initResources() override;
  void initSwapChainResources() override;
  void releaseSwapChainResources() override;
  void releaseResources() override;

  void startNextFrame() override;

protected:
  VkShaderModule createShader(const QString &name,
                              const shaderc_shader_kind kind);

  QVulkanWindow *m_window;
  QVulkanDeviceFunctions *m_devFuncs;

  VkDeviceMemory m_bufMem = VK_NULL_HANDLE;
  VkBuffer m_buf = VK_NULL_HANDLE;
  VkDescriptorBufferInfo
      m_uniformBufInfo[QVulkanWindow::MAX_CONCURRENT_FRAME_COUNT];

  VkDescriptorPool m_descPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet m_descSet[QVulkanWindow::MAX_CONCURRENT_FRAME_COUNT];

  VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;
  VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
  VkPipeline m_pipeline = VK_NULL_HANDLE;

  QMatrix4x4 m_proj;
  float m_rotation = 0.0f;
};

static float vertexData[] = { // Y up, front = CCW
    0.0f, 0.5f, 1.0f, 0.0f,  0.0f, -0.5f, -0.5f, 0.0f,
    1.0f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f,  1.0f};

static const int UNIFORM_DATA_SIZE = 16 * sizeof(float);

static inline VkDeviceSize aligned(VkDeviceSize v, VkDeviceSize byteAlign) {
  return (v + byteAlign - 1) & ~(byteAlign - 1);
}

TriangleRenderer::TriangleRenderer(QVulkanWindow *w, bool msaa) : m_window(w) {
  if (msaa) {
    const QList<int> counts = w->supportedSampleCounts();
    qDebug() << "Supported sample counts:" << counts;
    for (int s = 16; s >= 4; s /= 2) {
      if (counts.contains(s)) {
        qDebug("Requesting sample count %d", s);
        m_window->setSampleCount(s);
        break;
      }
    }
  }
}

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

VkShaderModule TriangleRenderer::createShader(const QString &name,
                                              shaderc_shader_kind kind) {
  auto blob = compile_file(name.toStdString(), kind);

  VkShaderModuleCreateInfo shaderInfo;
  memset(&shaderInfo, 0, sizeof(shaderInfo));
  shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderInfo.codeSize = blob.size() * sizeof(uint32_t);
  shaderInfo.pCode = reinterpret_cast<const uint32_t *>(blob.data());
  VkShaderModule shaderModule;
  VkResult err = m_devFuncs->vkCreateShaderModule(
      m_window->device(), &shaderInfo, nullptr, &shaderModule);
  if (err != VK_SUCCESS) {
    qWarning("Failed to create shader module: %d", err);
    return VK_NULL_HANDLE;
  }

  return shaderModule;
}

void TriangleRenderer::initResources() {
  qDebug("initResources");

  VkDevice dev = m_window->device();
  m_devFuncs = m_window->vulkanInstance()->deviceFunctions(dev);

  // Prepare the vertex and uniform data. The vertex data will never
  // change so one buffer is sufficient regardless of the value of
  // QVulkanWindow::CONCURRENT_FRAME_COUNT. Uniform data is changing per
  // frame however so active frames have to have a dedicated copy.

  // Use just one memory allocation and one buffer. We will then specify the
  // appropriate offsets for uniform buffers in the VkDescriptorBufferInfo.
  // Have to watch out for
  // VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment, though.

  // The uniform buffer is not strictly required in this example, we could
  // have used push constants as well since our single matrix (64 bytes) fits
  // into the spec mandated minimum limit of 128 bytes. However, once that
  // limit is not sufficient, the per-frame buffers, as shown below, will
  // become necessary.

  const int concurrentFrameCount = m_window->concurrentFrameCount();
  const VkPhysicalDeviceLimits *pdevLimits =
      &m_window->physicalDeviceProperties()->limits;
  const VkDeviceSize uniAlign = pdevLimits->minUniformBufferOffsetAlignment;
  qDebug("uniform buffer offset alignment is %u", (uint)uniAlign);
  VkBufferCreateInfo bufInfo;
  memset(&bufInfo, 0, sizeof(bufInfo));
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  // Our internal layout is vertex, uniform, uniform, ... with each uniform
  // buffer start offset aligned to uniAlign.
  const VkDeviceSize vertexAllocSize = aligned(sizeof(vertexData), uniAlign);
  const VkDeviceSize uniformAllocSize = aligned(UNIFORM_DATA_SIZE, uniAlign);
  bufInfo.size = vertexAllocSize + concurrentFrameCount * uniformAllocSize;
  bufInfo.usage =
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  VkResult err = m_devFuncs->vkCreateBuffer(dev, &bufInfo, nullptr, &m_buf);
  if (err != VK_SUCCESS)
    qFatal("Failed to create buffer: %d", err);

  VkMemoryRequirements memReq;
  m_devFuncs->vkGetBufferMemoryRequirements(dev, m_buf, &memReq);

  VkMemoryAllocateInfo memAllocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                       nullptr, memReq.size,
                                       m_window->hostVisibleMemoryIndex()};

  err = m_devFuncs->vkAllocateMemory(dev, &memAllocInfo, nullptr, &m_bufMem);
  if (err != VK_SUCCESS)
    qFatal("Failed to allocate memory: %d", err);

  err = m_devFuncs->vkBindBufferMemory(dev, m_buf, m_bufMem, 0);
  if (err != VK_SUCCESS)
    qFatal("Failed to bind buffer memory: %d", err);

  quint8 *p;
  err = m_devFuncs->vkMapMemory(dev, m_bufMem, 0, memReq.size, 0,
                                reinterpret_cast<void **>(&p));
  if (err != VK_SUCCESS)
    qFatal("Failed to map memory: %d", err);
  memcpy(p, vertexData, sizeof(vertexData));
  QMatrix4x4 ident;
  memset(m_uniformBufInfo, 0, sizeof(m_uniformBufInfo));
  for (int i = 0; i < concurrentFrameCount; ++i) {
    const VkDeviceSize offset = vertexAllocSize + i * uniformAllocSize;
    memcpy(p + offset, ident.constData(), 16 * sizeof(float));
    m_uniformBufInfo[i].buffer = m_buf;
    m_uniformBufInfo[i].offset = offset;
    m_uniformBufInfo[i].range = uniformAllocSize;
  }
  m_devFuncs->vkUnmapMemory(dev, m_bufMem);

  VkVertexInputBindingDescription vertexBindingDesc = {
      0, // binding
      5 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription vertexAttrDesc[] = {
      {   // position
       0, // location
       0, // binding
       VK_FORMAT_R32G32_SFLOAT, 0},
      {// color
       1, 0, VK_FORMAT_R32G32B32_SFLOAT, 2 * sizeof(float)}};

  VkPipelineVertexInputStateCreateInfo vertexInputInfo;
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.pNext = nullptr;
  vertexInputInfo.flags = 0;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &vertexBindingDesc;
  vertexInputInfo.vertexAttributeDescriptionCount = 2;
  vertexInputInfo.pVertexAttributeDescriptions = vertexAttrDesc;

  // Set up descriptor set and its layout.
  VkDescriptorPoolSize descPoolSizes = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                        uint32_t(concurrentFrameCount)};
  VkDescriptorPoolCreateInfo descPoolInfo;
  memset(&descPoolInfo, 0, sizeof(descPoolInfo));
  descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descPoolInfo.maxSets = concurrentFrameCount;
  descPoolInfo.poolSizeCount = 1;
  descPoolInfo.pPoolSizes = &descPoolSizes;
  err = m_devFuncs->vkCreateDescriptorPool(dev, &descPoolInfo, nullptr,
                                           &m_descPool);
  if (err != VK_SUCCESS)
    qFatal("Failed to create descriptor pool: %d", err);

  VkDescriptorSetLayoutBinding layoutBinding = {
      0, // binding
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,
      nullptr};
  VkDescriptorSetLayoutCreateInfo descLayoutInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1,
      &layoutBinding};
  err = m_devFuncs->vkCreateDescriptorSetLayout(dev, &descLayoutInfo, nullptr,
                                                &m_descSetLayout);
  if (err != VK_SUCCESS)
    qFatal("Failed to create descriptor set layout: %d", err);

  for (int i = 0; i < concurrentFrameCount; ++i) {
    VkDescriptorSetAllocateInfo descSetAllocInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, m_descPool, 1,
        &m_descSetLayout};
    err = m_devFuncs->vkAllocateDescriptorSets(dev, &descSetAllocInfo,
                                               &m_descSet[i]);
    if (err != VK_SUCCESS)
      qFatal("Failed to allocate descriptor set: %d", err);

    VkWriteDescriptorSet descWrite;
    memset(&descWrite, 0, sizeof(descWrite));
    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descWrite.dstSet = m_descSet[i];
    descWrite.descriptorCount = 1;
    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descWrite.pBufferInfo = &m_uniformBufInfo[i];
    m_devFuncs->vkUpdateDescriptorSets(dev, 1, &descWrite, 0, nullptr);
  }

  // Pipeline cache
  VkPipelineCacheCreateInfo pipelineCacheInfo;
  memset(&pipelineCacheInfo, 0, sizeof(pipelineCacheInfo));
  pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  err = m_devFuncs->vkCreatePipelineCache(dev, &pipelineCacheInfo, nullptr,
                                          &m_pipelineCache);
  if (err != VK_SUCCESS)
    qFatal("Failed to create pipeline cache: %d", err);

  // Pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutInfo;
  memset(&pipelineLayoutInfo, 0, sizeof(pipelineLayoutInfo));
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &m_descSetLayout;
  err = m_devFuncs->vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr,
                                           &m_pipelineLayout);
  if (err != VK_SUCCESS)
    qFatal("Failed to create pipeline layout: %d", err);

  // Shaders
  VkShaderModule vertShaderModule =
      createShader(QStringLiteral("./main.vert"), shaderc_glsl_vertex_shader);
  VkShaderModule fragShaderModule =
      createShader(QStringLiteral("./main.frag"), shaderc_glsl_fragment_shader);

  // Graphics pipeline
  VkGraphicsPipelineCreateInfo pipelineInfo;
  memset(&pipelineInfo, 0, sizeof(pipelineInfo));
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

  VkPipelineShaderStageCreateInfo shaderStages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
       VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule, "main", nullptr},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
       VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule, "main", nullptr}};
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;

  pipelineInfo.pVertexInputState = &vertexInputInfo;

  VkPipelineInputAssemblyStateCreateInfo ia;
  memset(&ia, 0, sizeof(ia));
  ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  pipelineInfo.pInputAssemblyState = &ia;

  // The viewport and scissor will be set dynamically via
  // vkCmdSetViewport/Scissor. This way the pipeline does not need to be touched
  // when resizing the window.
  VkPipelineViewportStateCreateInfo vp;
  memset(&vp, 0, sizeof(vp));
  vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  pipelineInfo.pViewportState = &vp;

  VkPipelineRasterizationStateCreateInfo rs;
  memset(&rs, 0, sizeof(rs));
  rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE; // we want the back face as well
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  pipelineInfo.pRasterizationState = &rs;

  VkPipelineMultisampleStateCreateInfo ms;
  memset(&ms, 0, sizeof(ms));
  ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  // Enable multisampling.
  ms.rasterizationSamples = m_window->sampleCountFlagBits();
  pipelineInfo.pMultisampleState = &ms;

  VkPipelineDepthStencilStateCreateInfo ds;
  memset(&ds, 0, sizeof(ds));
  ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  ds.depthTestEnable = VK_TRUE;
  ds.depthWriteEnable = VK_TRUE;
  ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  pipelineInfo.pDepthStencilState = &ds;

  VkPipelineColorBlendStateCreateInfo cb;
  memset(&cb, 0, sizeof(cb));
  cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  // no blend, write out all of rgba
  VkPipelineColorBlendAttachmentState att;
  memset(&att, 0, sizeof(att));
  att.colorWriteMask = 0xF;
  cb.attachmentCount = 1;
  cb.pAttachments = &att;
  pipelineInfo.pColorBlendState = &cb;

  VkDynamicState dynEnable[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dyn;
  memset(&dyn, 0, sizeof(dyn));
  dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dyn.dynamicStateCount = sizeof(dynEnable) / sizeof(VkDynamicState);
  dyn.pDynamicStates = dynEnable;
  pipelineInfo.pDynamicState = &dyn;

  pipelineInfo.layout = m_pipelineLayout;
  pipelineInfo.renderPass = m_window->defaultRenderPass();

  err = m_devFuncs->vkCreateGraphicsPipelines(
      dev, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_pipeline);
  if (err != VK_SUCCESS)
    qFatal("Failed to create graphics pipeline: %d", err);

  if (vertShaderModule)
    m_devFuncs->vkDestroyShaderModule(dev, vertShaderModule, nullptr);
  if (fragShaderModule)
    m_devFuncs->vkDestroyShaderModule(dev, fragShaderModule, nullptr);
}

void TriangleRenderer::initSwapChainResources() {
  qDebug("initSwapChainResources");

  // Projection matrix
  m_proj = m_window->clipCorrectionMatrix(); // adjust for Vulkan-OpenGL clip
                                             // space differences
  const QSize sz = m_window->swapChainImageSize();
  m_proj.perspective(45.0f, sz.width() / (float)sz.height(), 0.01f, 100.0f);
  m_proj.translate(0, 0, -4);
}

void TriangleRenderer::releaseSwapChainResources() {
  qDebug("releaseSwapChainResources");
}

void TriangleRenderer::releaseResources() {
  qDebug("releaseResources");

  VkDevice dev = m_window->device();

  if (m_pipeline) {
    m_devFuncs->vkDestroyPipeline(dev, m_pipeline, nullptr);
    m_pipeline = VK_NULL_HANDLE;
  }

  if (m_pipelineLayout) {
    m_devFuncs->vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
    m_pipelineLayout = VK_NULL_HANDLE;
  }

  if (m_pipelineCache) {
    m_devFuncs->vkDestroyPipelineCache(dev, m_pipelineCache, nullptr);
    m_pipelineCache = VK_NULL_HANDLE;
  }

  if (m_descSetLayout) {
    m_devFuncs->vkDestroyDescriptorSetLayout(dev, m_descSetLayout, nullptr);
    m_descSetLayout = VK_NULL_HANDLE;
  }

  if (m_descPool) {
    m_devFuncs->vkDestroyDescriptorPool(dev, m_descPool, nullptr);
    m_descPool = VK_NULL_HANDLE;
  }

  if (m_buf) {
    m_devFuncs->vkDestroyBuffer(dev, m_buf, nullptr);
    m_buf = VK_NULL_HANDLE;
  }

  if (m_bufMem) {
    m_devFuncs->vkFreeMemory(dev, m_bufMem, nullptr);
    m_bufMem = VK_NULL_HANDLE;
  }
}

void TriangleRenderer::startNextFrame() {
  VkDevice dev = m_window->device();
  VkCommandBuffer cb = m_window->currentCommandBuffer();
  const QSize sz = m_window->swapChainImageSize();

  VkClearColorValue clearColor = {{0, 0, 0, 1}};
  VkClearDepthStencilValue clearDS = {1, 0};
  VkClearValue clearValues[3];
  memset(clearValues, 0, sizeof(clearValues));
  clearValues[0].color = clearValues[2].color = clearColor;
  clearValues[1].depthStencil = clearDS;

  VkRenderPassBeginInfo rpBeginInfo;
  memset(&rpBeginInfo, 0, sizeof(rpBeginInfo));
  rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rpBeginInfo.renderPass = m_window->defaultRenderPass();
  rpBeginInfo.framebuffer = m_window->currentFramebuffer();
  rpBeginInfo.renderArea.extent.width = sz.width();
  rpBeginInfo.renderArea.extent.height = sz.height();
  rpBeginInfo.clearValueCount =
      m_window->sampleCountFlagBits() > VK_SAMPLE_COUNT_1_BIT ? 3 : 2;
  rpBeginInfo.pClearValues = clearValues;
  VkCommandBuffer cmdBuf = m_window->currentCommandBuffer();
  m_devFuncs->vkCmdBeginRenderPass(cmdBuf, &rpBeginInfo,
                                   VK_SUBPASS_CONTENTS_INLINE);

  quint8 *p;
  VkResult err = m_devFuncs->vkMapMemory(
      dev, m_bufMem, m_uniformBufInfo[m_window->currentFrame()].offset,
      UNIFORM_DATA_SIZE, 0, reinterpret_cast<void **>(&p));
  if (err != VK_SUCCESS)
    qFatal("Failed to map memory: %d", err);
  QMatrix4x4 m = m_proj;
  m.rotate(m_rotation, 0, 1, 0);
  memcpy(p, m.constData(), 16 * sizeof(float));
  m_devFuncs->vkUnmapMemory(dev, m_bufMem);

  // Not exactly a real animation system, just advance on every frame for now.
  m_rotation += 1.0f;

  m_devFuncs->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_pipeline);
  m_devFuncs->vkCmdBindDescriptorSets(
      cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1,
      &m_descSet[m_window->currentFrame()], 0, nullptr);
  VkDeviceSize vbOffset = 0;
  m_devFuncs->vkCmdBindVertexBuffers(cb, 0, 1, &m_buf, &vbOffset);

  VkViewport viewport;
  viewport.x = viewport.y = 0;
  viewport.width = sz.width();
  viewport.height = sz.height();
  viewport.minDepth = 0;
  viewport.maxDepth = 1;
  m_devFuncs->vkCmdSetViewport(cb, 0, 1, &viewport);

  VkRect2D scissor;
  scissor.offset.x = scissor.offset.y = 0;
  scissor.extent.width = viewport.width;
  scissor.extent.height = viewport.height;
  m_devFuncs->vkCmdSetScissor(cb, 0, 1, &scissor);

  m_devFuncs->vkCmdDraw(cb, 3, 1, 0, 0);

  m_devFuncs->vkCmdEndRenderPass(cmdBuf);

  m_window->frameReady();
  m_window->requestUpdate(); // render continuously, throttled by the
                             // presentation rate
}

class VulkanRenderer : public TriangleRenderer {
public:
  VulkanRenderer(QVulkanWindow *w);

  void initResources() override;
  void startNextFrame() override;
  void reinitializeResources();
};

void VulkanRenderer::reinitializeResources() {
  releaseResources();
  initResources();
}

class VulkanWindow : public QVulkanWindow {
  Q_OBJECT

public:
  QVulkanWindowRenderer *createRenderer() override;

  VulkanRenderer *m_renderer;

signals:
  void vulkanInfoReceived(const QString &text);
  void frameQueued(int colorValue);

public slots:
  void onReinitializeResources();
};

void VulkanWindow::onReinitializeResources() {
  m_renderer->reinitializeResources();
}

class MainWindow : public QWidget {
  Q_OBJECT

public:
  explicit MainWindow(VulkanWindow *w, QPlainTextEdit *logWidget);
  void editShaderFile(const QString &filePath);

public slots:
  void onShaderEditAccepted();
  void onVulkanInfoReceived(const QString &text);
  void onFrameQueued(int colorValue);
  void onGrabRequested();

private:
  VulkanWindow *m_window;
  QTabWidget *m_infoTab;
  QPlainTextEdit *m_info;
  QLCDNumber *m_number;
  QTextEdit *m_shaderEditor;
  QString m_currentShaderPath;
};

MainWindow::MainWindow(VulkanWindow *w, QPlainTextEdit *logWidget)
    : m_window(w) {
  QWidget *wrapper = QWidget::createWindowContainer(w);

  m_info = new QPlainTextEdit;
  m_info->setReadOnly(true);

  m_number = new QLCDNumber(3);
  m_number->setSegmentStyle(QLCDNumber::Filled);

  QPushButton *grabButton = new QPushButton(tr("&Grab"));
  grabButton->setFocusPolicy(Qt::NoFocus);

  connect(grabButton, &QPushButton::clicked, this,
          &MainWindow::onGrabRequested);

  QPushButton *editVertShaderButton = new QPushButton(tr("Edit main.vert"));
  connect(editVertShaderButton, &QPushButton::clicked, this,
          [this]() { editShaderFile(QStringLiteral("./main.vert")); });

  QPushButton *editFragShaderButton = new QPushButton(tr("Edit main.frag"));
  connect(editFragShaderButton, &QPushButton::clicked, this,
          [this]() { editShaderFile(QStringLiteral("./main.frag")); });

  QPushButton *compileButton = new QPushButton(tr("&Compile"));
  compileButton->setFocusPolicy(Qt::NoFocus);

  connect(compileButton, &QPushButton::clicked, m_window,
          &VulkanWindow::onReinitializeResources);

  QPushButton *quitButton = new QPushButton(tr("&Quit"));
  quitButton->setFocusPolicy(Qt::NoFocus);

  connect(quitButton, &QPushButton::clicked, qApp, &QCoreApplication::quit);

  QVBoxLayout *layout = new QVBoxLayout;
  m_infoTab = new QTabWidget(this);
  m_infoTab->addTab(m_info, tr("Vulkan Info"));
  m_infoTab->addTab(logWidget, tr("Debug Log"));
  layout->addWidget(m_infoTab, 2);
  layout->addWidget(m_number, 1);
  layout->addWidget(wrapper, 5);
  layout->addWidget(grabButton, 1);
  layout->addWidget(editVertShaderButton, 1);
  layout->addWidget(editFragShaderButton, 1);
  layout->addWidget(compileButton, 1);
  layout->addWidget(quitButton, 1);
  setLayout(layout);
}

void MainWindow::editShaderFile(const QString &filePath) {
  QFile file(filePath);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    return;

  QTextStream in(&file);
  QString shaderCode = in.readAll();
  file.close();

  QDialog dialog(this);
  QVBoxLayout layout(&dialog);

  m_shaderEditor = new QTextEdit;
  m_shaderEditor->setPlainText(shaderCode);
  layout.addWidget(m_shaderEditor);

  QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  connect(&buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(&buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
  layout.addWidget(&buttonBox);

  m_currentShaderPath = filePath;
  if (dialog.exec() == QDialog::Accepted) {
    onShaderEditAccepted();
  }
}

void MainWindow::onShaderEditAccepted() {
  QFile file(m_currentShaderPath);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    return;

  QTextStream out(&file);
  out << m_shaderEditor->toPlainText();
  file.close();

  m_window->onReinitializeResources(); // Trigger recompilation
}

void MainWindow::onVulkanInfoReceived(const QString &text) {
  m_info->setPlainText(text);
}

void MainWindow::onFrameQueued(int colorValue) {
  m_number->display(colorValue);
}

void MainWindow::onGrabRequested() {
  if (!m_window->supportsGrab()) {
    QMessageBox::warning(this, tr("Cannot grab"),
                         tr("This swapchain does not support readbacks."));
    return;
  }

  QImage img = m_window->grab();

  // Our startNextFrame() implementation is synchronous so img is ready to be
  // used right here.

  QFileDialog fd(this);
  fd.setAcceptMode(QFileDialog::AcceptSave);
  fd.setDefaultSuffix("png");
  fd.selectFile("test.png");
  if (fd.exec() == QDialog::Accepted)
    img.save(fd.selectedFiles().first());
}

QVulkanWindowRenderer *VulkanWindow::createRenderer() {
  m_renderer = new VulkanRenderer(this);
  return m_renderer;
}

VulkanRenderer::VulkanRenderer(QVulkanWindow *w) : TriangleRenderer(w) {}

void VulkanRenderer::initResources() {
  TriangleRenderer::initResources();

  QVulkanInstance *inst = m_window->vulkanInstance();
  m_devFuncs = inst->deviceFunctions(m_window->device());

  QString info;
  info += QString::asprintf("Number of physical devices: %d\n",
                            int(m_window->availablePhysicalDevices().count()));

  QVulkanFunctions *f = inst->functions();
  VkPhysicalDeviceProperties props;
  f->vkGetPhysicalDeviceProperties(m_window->physicalDevice(), &props);
  info += QString::asprintf(
      "Active physical device name: '%s' version %d.%d.%d\nAPI version "
      "%d.%d.%d\n",
      props.deviceName, VK_VERSION_MAJOR(props.driverVersion),
      VK_VERSION_MINOR(props.driverVersion),
      VK_VERSION_PATCH(props.driverVersion), VK_VERSION_MAJOR(props.apiVersion),
      VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));

  info += QStringLiteral("Supported instance layers:\n");
  for (const QVulkanLayer &layer : inst->supportedLayers())
    info += QString::asprintf("    %s v%u\n", layer.name.constData(),
                              layer.version);
  info += QStringLiteral("Enabled instance layers:\n");
  for (const QByteArray &layer : inst->layers())
    info += QString::asprintf("    %s\n", layer.constData());

  info += QStringLiteral("Supported instance extensions:\n");
  for (const QVulkanExtension &ext : inst->supportedExtensions())
    info +=
        QString::asprintf("    %s v%u\n", ext.name.constData(), ext.version);
  info += QStringLiteral("Enabled instance extensions:\n");
  for (const QByteArray &ext : inst->extensions())
    info += QString::asprintf("    %s\n", ext.constData());

  info += QString::asprintf("Color format: %u\nDepth-stencil format: %u\n",
                            m_window->colorFormat(),
                            m_window->depthStencilFormat());

  info += QStringLiteral("Supported sample counts:");
  const QList<int> sampleCounts = m_window->supportedSampleCounts();
  for (int count : sampleCounts)
    info += QLatin1Char(' ') + QString::number(count);
  info += QLatin1Char('\n');

  emit static_cast<VulkanWindow *>(m_window)->vulkanInfoReceived(info);
}

void VulkanRenderer::startNextFrame() {
  TriangleRenderer::startNextFrame();
  emit static_cast<VulkanWindow *>(m_window)->frameQueued(int(m_rotation) %
                                                          360);
}

Q_LOGGING_CATEGORY(lcVk, "qt.vulkan")

static QPointer<QPlainTextEdit> messageLogWidget;
static QtMessageHandler oldMessageHandler = nullptr;

static void messageHandler(QtMsgType msgType,
                           const QMessageLogContext &logContext,
                           const QString &text) {
  if (!messageLogWidget.isNull())
    messageLogWidget->appendPlainText(text);
  if (oldMessageHandler)
    oldMessageHandler(msgType, logContext, text);
}

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);

  messageLogWidget = new QPlainTextEdit(QLatin1String(QLibraryInfo::build()) +
                                        QLatin1Char('\n'));
  messageLogWidget->setReadOnly(true);

  oldMessageHandler = qInstallMessageHandler(messageHandler);

  QLoggingCategory::setFilterRules(QStringLiteral("qt.vulkan=true"));

  QVulkanInstance inst;
  inst.setLayers({"VK_LAYER_KHRONOS_validation"});

  if (!inst.create())
    qFatal("Failed to create Vulkan instance: %d", inst.errorCode());

  VulkanWindow *vulkanWindow = new VulkanWindow;
  vulkanWindow->setVulkanInstance(&inst);

  MainWindow mainWindow(vulkanWindow, messageLogWidget.data());
  QObject::connect(vulkanWindow, &VulkanWindow::vulkanInfoReceived, &mainWindow,
                   &MainWindow::onVulkanInfoReceived);
  QObject::connect(vulkanWindow, &VulkanWindow::frameQueued, &mainWindow,
                   &MainWindow::onFrameQueued);

  mainWindow.resize(1024, 768);
  mainWindow.show();

  return app.exec();
}

#include "main.moc"
