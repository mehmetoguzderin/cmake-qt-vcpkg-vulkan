#version 460

layout(std140, binding = 0) uniform buf {
  mat4 mvp;
  mat4 inverse_mvp;
  vec4 origin_0_0;
  vec4 origin_1_0;
  vec4 origin_0_1;
  vec4 direction_0_0;
  vec4 direction_1_0;
  vec4 direction_0_1;
  vec4 data_width_height_time;
}
ubuf;

layout(binding = 1) uniform texture2D textureImage;
layout(binding = 2) uniform sampler textureSampler;

layout(location = 0) in vec3 v_uv;

layout(location = 0) out vec4 fragColor;

void main() {
  fragColor = texture(sampler2D(textureImage, textureSampler),
                      gl_FragCoord.xy / ubuf.data_width_height_time.xy);
}
