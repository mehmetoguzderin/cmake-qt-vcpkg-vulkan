#include "main.hpp"

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

  VkImage m_textureImage = VK_NULL_HANDLE;
  VkDeviceMemory m_textureImageMemory = VK_NULL_HANDLE;
  VkImageView m_textureImageView = VK_NULL_HANDLE;
  VkSampler m_textureSampler = VK_NULL_HANDLE;

  VkDescriptorPool m_descPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet m_descSet[QVulkanWindow::MAX_CONCURRENT_FRAME_COUNT];

  VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;
  VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
  VkPipeline m_pipeline = VK_NULL_HANDLE;

  VkDescriptorPool m_computeDescPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_computeDescSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet m_computeDescSet;
  VkPipelineLayout m_computePipelineLayout = VK_NULL_HANDLE;
  VkPipeline m_computePipeline = VK_NULL_HANDLE;

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
  QVulkanInfoVector<QVulkanExtension> supportedExtensions =
      m_window->supportedDeviceExtensions();
  if (supportedExtensions.contains(
          QByteArrayLiteral("VK_KHR_portability_subset"))) {
    qDebug("Enabling VK_KHR_portability_subset");
    QByteArrayList extensions;
    extensions.append(QByteArrayLiteral("VK_KHR_portability_subset"));
    m_window->setDeviceExtensions(extensions);
  }
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

  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.anisotropyEnable = VK_TRUE;
  samplerInfo.maxAnisotropy = 16;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (m_devFuncs->vkCreateSampler(m_window->device(), &samplerInfo, nullptr,
                                  &m_textureSampler) != VK_SUCCESS) {
    qFatal("failed to create texture sampler");
  }

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
  std::array<VkDescriptorPoolSize, 3> poolSizes{};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = concurrentFrameCount;

  poolSizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  poolSizes[1].descriptorCount = concurrentFrameCount;

  poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLER;
  poolSizes[2].descriptorCount = concurrentFrameCount;

  VkDescriptorPoolCreateInfo descPoolInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      nullptr,
      0,
      static_cast<uint32_t>(concurrentFrameCount),
      static_cast<uint32_t>(poolSizes.size()),
      poolSizes.data()};

  err = m_devFuncs->vkCreateDescriptorPool(dev, &descPoolInfo, nullptr,
                                           &m_descPool);
  if (err != VK_SUCCESS)
    qFatal("Failed to create descriptor pool: %d", err);

  std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings{};

  // Uniform buffer binding
  layoutBindings[0].binding = 0;
  layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  layoutBindings[0].descriptorCount = 1;
  layoutBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  // Texture binding
  layoutBindings[1].binding = 1;
  layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  layoutBindings[1].descriptorCount = 1;
  layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  // Sampler binding
  layoutBindings[2].binding = 2;
  layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  layoutBindings[2].descriptorCount = 1;
  layoutBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo descLayoutInfo = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
      static_cast<uint32_t>(layoutBindings.size()), layoutBindings.data()};
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

  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  poolSize.descriptorCount = 1; // Adjust if you have more images

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = 1; // Adjust if you have more descriptor sets

  if (m_devFuncs->vkCreateDescriptorPool(dev, &poolInfo, nullptr,
                                         &m_computeDescPool) != VK_SUCCESS) {
    qFatal("Failed to create descriptor pool for compute shader");
  }

  VkDescriptorSetLayoutBinding samplerLayoutBinding{};
  samplerLayoutBinding.binding = 0;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  samplerLayoutBinding.pImmutableSamplers = nullptr;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &samplerLayoutBinding;

  if (m_devFuncs->vkCreateDescriptorSetLayout(
          dev, &layoutInfo, nullptr, &m_computeDescSetLayout) != VK_SUCCESS) {
    qFatal("Failed to create descriptor set layout for compute shader");
  }

  // allocate compute descriptor set
  VkDescriptorSetAllocateInfo computeAllocInfo{};
  computeAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  computeAllocInfo.descriptorPool = m_computeDescPool;
  computeAllocInfo.descriptorSetCount = 1;
  computeAllocInfo.pSetLayouts = &m_computeDescSetLayout;

  if (m_devFuncs->vkAllocateDescriptorSets(dev, &computeAllocInfo,
                                           &m_computeDescSet) != VK_SUCCESS) {
    qFatal("Failed to allocate descriptor sets for compute shader");
  }

  VkPipelineLayoutCreateInfo computePipelineLayoutInfo{};
  memset(&computePipelineLayoutInfo, 0, sizeof(computePipelineLayoutInfo));
  computePipelineLayoutInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  computePipelineLayoutInfo.setLayoutCount = 1;
  computePipelineLayoutInfo.pSetLayouts = &m_computeDescSetLayout;

  if (m_devFuncs->vkCreatePipelineLayout(dev, &computePipelineLayoutInfo,
                                         nullptr, &m_computePipelineLayout) !=
      VK_SUCCESS) {
    qFatal("Failed to create compute pipeline layout");
  }

  VkShaderModule compShaderModule =
      createShader(QStringLiteral("./main.comp"), shaderc_compute_shader);

  VkPipelineShaderStageCreateInfo compShaderStageInfo{};
  compShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  compShaderStageInfo.module = compShaderModule;
  compShaderStageInfo.pName = "main";

  VkComputePipelineCreateInfo compPipelineCreateInfo{};
  compPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  compPipelineCreateInfo.stage = compShaderStageInfo;
  compPipelineCreateInfo.layout = m_computePipelineLayout;

  err = m_devFuncs->vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1,
                                             &compPipelineCreateInfo, nullptr,
                                             &m_computePipeline);
  if (err != VK_SUCCESS)
    qFatal("Failed to create compute pipeline: %d", err);

  if (compShaderModule)
    m_devFuncs->vkDestroyShaderModule(dev, compShaderModule, nullptr);
}

void TriangleRenderer::initSwapChainResources() {
  qDebug("initSwapChainResources");

  // Projection matrix
  m_proj = m_window->clipCorrectionMatrix(); // adjust for Vulkan-OpenGL clip
                                             // space differences
  const QSize sz = m_window->swapChainImageSize();
  m_proj.perspective(45.0f, sz.width() / (float)sz.height(), 0.01f, 100.0f);
  m_proj.translate(0, 0, -4);

  VkDevice dev = m_window->device();
  m_devFuncs = m_window->vulkanInstance()->deviceFunctions(dev);
  m_devFuncs->vkDeviceWaitIdle(dev);

  const int concurrentFrameCount = m_window->concurrentFrameCount();

  VkResult err = VK_SUCCESS;

  VkExtent3D textureExtent = {
      static_cast<uint32_t>(sz.width()), static_cast<uint32_t>(sz.height()),
      1 // depth
  };

  // print extent
  qDebug("texture extent: %d x %d x %d", textureExtent.width,
         textureExtent.height, textureExtent.depth);

  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent = textureExtent;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (m_devFuncs->vkCreateImage(dev, &imageInfo, nullptr, &m_textureImage) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  VkMemoryRequirements memRequirements;
  m_devFuncs->vkGetImageMemoryRequirements(dev, m_textureImage,
                                           &memRequirements);
  VkPhysicalDeviceMemoryProperties memProperties;
  m_window->vulkanInstance()->functions()->vkGetPhysicalDeviceMemoryProperties(
      m_window->physicalDevice(), &memProperties);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = UINT32_MAX;
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
      allocInfo.memoryTypeIndex = i;
      break;
    }
  }
  if (allocInfo.memoryTypeIndex == UINT32_MAX) {
    qFatal("failed to find suitable memory type");
  }

  if (m_devFuncs->vkAllocateMemory(dev, &allocInfo, nullptr,
                                   &m_textureImageMemory) != VK_SUCCESS) {
    qFatal("failed to allocate image memory");
  }

  m_devFuncs->vkBindImageMemory(dev, m_textureImage, m_textureImageMemory, 0);

  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = m_textureImage;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (m_devFuncs->vkCreateImageView(dev, &viewInfo, nullptr,
                                    &m_textureImageView) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture image view!");
  }

  for (int i = 0; i < concurrentFrameCount; ++i) {
    std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

    // Uniform buffer
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = m_descSet[i];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &m_uniformBufInfo[i];

    // Texture image
    VkDescriptorImageInfo textureImageInfo{};
    textureImageInfo.imageView = m_textureImageView;
    textureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = m_descSet[i];
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pImageInfo = &textureImageInfo;

    // Sampler
    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = m_textureSampler;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = m_descSet[i];
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].dstArrayElement = 0;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pImageInfo = &samplerInfo;

    m_devFuncs->vkUpdateDescriptorSets(dev, descriptorWrites.size(),
                                       descriptorWrites.data(), 0, nullptr);
  }

  {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = m_textureImageView; // replace with your image view
    imageInfo.imageLayout =
        VK_IMAGE_LAYOUT_GENERAL; // Layout used in the compute shader

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = m_computeDescSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &imageInfo;

    m_devFuncs->vkUpdateDescriptorSets(dev, 1, &descriptorWrite, 0, nullptr);
  }
}

void TriangleRenderer::releaseSwapChainResources() {
  qDebug("releaseSwapChainResources");

  if (m_textureImageView != VK_NULL_HANDLE) {
    m_devFuncs->vkDestroyImageView(m_window->device(), m_textureImageView,
                                   nullptr);
    m_textureImageView = VK_NULL_HANDLE;
  }

  if (m_textureImage != VK_NULL_HANDLE) {
    m_devFuncs->vkDestroyImage(m_window->device(), m_textureImage, nullptr);
    m_textureImage = VK_NULL_HANDLE;
  }

  if (m_textureImageMemory != VK_NULL_HANDLE) {
    m_devFuncs->vkFreeMemory(m_window->device(), m_textureImageMemory, nullptr);
    m_textureImageMemory = VK_NULL_HANDLE;
  }
}

void TriangleRenderer::releaseResources() {
  qDebug("releaseResources");

  VkDevice dev = m_window->device();

  if (m_computePipeline) {
    m_devFuncs->vkDestroyPipeline(dev, m_computePipeline, nullptr);
    m_computePipeline = VK_NULL_HANDLE;
  }

  if (m_computePipelineLayout) {
    m_devFuncs->vkDestroyPipelineLayout(dev, m_computePipelineLayout, nullptr);
    m_computePipelineLayout = VK_NULL_HANDLE;
  }

  if (m_computeDescSetLayout) {
    m_devFuncs->vkDestroyDescriptorSetLayout(dev, m_computeDescSetLayout,
                                             nullptr);
    m_computeDescSetLayout = VK_NULL_HANDLE;
  }

  if (m_computeDescPool) {
    m_devFuncs->vkDestroyDescriptorPool(dev, m_computeDescPool, nullptr);
    m_computeDescPool = VK_NULL_HANDLE;
  }

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

  if (m_textureSampler != VK_NULL_HANDLE) {
    m_devFuncs->vkDestroySampler(m_window->device(), m_textureSampler, nullptr);
    m_textureSampler = VK_NULL_HANDLE;
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
  m_devFuncs->vkDeviceWaitIdle(dev);
  VkCommandBuffer cb = m_window->currentCommandBuffer();
  QSize sz = m_window->swapChainImageSize();

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = m_window->graphicsCommandPool();
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer computeCommandBuffer;
  m_devFuncs->vkAllocateCommandBuffers(dev, &allocInfo, &computeCommandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  m_devFuncs->vkBeginCommandBuffer(computeCommandBuffer, &beginInfo);

  // Image layout transition barrier
  VkImageMemoryBarrier computeBarrier{};
  computeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  computeBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  computeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  computeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  computeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  computeBarrier.image = m_textureImage;
  computeBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  computeBarrier.subresourceRange.baseMipLevel = 0;
  computeBarrier.subresourceRange.levelCount = 1;
  computeBarrier.subresourceRange.baseArrayLayer = 0;
  computeBarrier.subresourceRange.layerCount = 1;
  computeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  computeBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

  m_devFuncs->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   0,          // No dependency flags
                                   0, nullptr, // No memory barriers
                                   0, nullptr, // No buffer memory barriers
                                   1, &computeBarrier // Image memory barrier
  );

  // Dispatch compute shader
  m_devFuncs->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_computePipeline);

  // Bind descriptor set
  m_devFuncs->vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      m_computePipelineLayout, 0, 1,
                                      &m_computeDescSet, 0, nullptr);

  m_devFuncs->vkCmdDispatch(cb, (uint32_t)ceil(sz.width() / float(8)),
                            (uint32_t)ceil(sz.height() / float(8)), 1);

  // End compute command buffer
  m_devFuncs->vkEndCommandBuffer(computeCommandBuffer);

  // Submit compute command buffer and wait for completion
  VkSubmitInfo computeSubmitInfo{};
  computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  computeSubmitInfo.commandBufferCount = 1;
  computeSubmitInfo.pCommandBuffers = &computeCommandBuffer;

  VkQueue computeQueue;
  m_devFuncs->vkGetDeviceQueue(dev, 0, 0, &computeQueue);
  m_devFuncs->vkQueueSubmit(computeQueue, 1, &computeSubmitInfo,
                            VK_NULL_HANDLE);
  m_devFuncs->vkQueueWaitIdle(computeQueue);

  m_devFuncs->vkDeviceWaitIdle(dev);

  // Image layout transition barrier
  VkImageMemoryBarrier renderBarrier{};
  renderBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  renderBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  renderBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  renderBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  renderBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  renderBarrier.image = m_textureImage;
  renderBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  renderBarrier.subresourceRange.baseMipLevel = 0;
  renderBarrier.subresourceRange.levelCount = 1;
  renderBarrier.subresourceRange.baseArrayLayer = 0;
  renderBarrier.subresourceRange.layerCount = 1;
  renderBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  renderBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  // Apply the barrier
  m_devFuncs->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   0,          // No dependency flags
                                   0, nullptr, // No memory barriers
                                   0, nullptr, // No buffer memory barriers
                                   1, &renderBarrier // Image memory barrier
  );

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
  m_devFuncs->vkDeviceWaitIdle(dev);
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
  m_window->vulkanInstance()
      ->deviceFunctions(m_window->device())
      ->vkDeviceWaitIdle(m_window->device());
  releaseSwapChainResources();
  releaseResources();
  initResources();
  initSwapChainResources();
}

class VulkanWindow : public QVulkanWindow {
  Q_OBJECT

public:
  QVulkanWindowRenderer *createRenderer() override;

  VulkanRenderer *m_renderer;

  void keyPressEvent(QKeyEvent *event) override;

signals:
  void vulkanInfoReceived(const QString &text);
  void frameQueued(int colorValue);

public slots:
  void onReinitializeResources();
};

void VulkanWindow::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key_Escape) {
    qApp->quit();
  } else {
    QVulkanWindow::keyPressEvent(event);
  }
}

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

  QPushButton *editCompShaderButton = new QPushButton(tr("Edit main.comp"));
  connect(editCompShaderButton, &QPushButton::clicked, this,
          [this]() { editShaderFile(QStringLiteral("./main.comp")); });

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
  layout->addWidget(editCompShaderButton, 1);
  layout->addWidget(compileButton, 1);
  layout->addWidget(quitButton, 1);
  setLayout(layout);

  wrapper->setFocus();
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
  inst.setApiVersion(QVersionNumber(1, 1));
  inst.setExtensions({"VK_KHR_get_physical_device_properties2"});
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
