#version 460

layout(binding = 1) uniform texture2D textureImage;
layout(binding = 2) uniform sampler textureSampler;

layout(location = 0) in vec3 v_uv;

layout(location = 0) out vec4 fragColor;

void main() {
  fragColor = texture(sampler2D(textureImage, textureSampler), v_uv.xy) +
              vec4(v_uv.xy, 0.0, 1.0);
}
