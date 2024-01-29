#version 460

layout(std140, binding = 0) uniform buf {
  mat4 mvp;
  vec4 data;
}
ubuf;

layout(location = 0) out vec3 v_color;

out gl_PerVertex { vec4 gl_Position; };

// #define MVP_TRIANGLE

#ifdef MVP_TRIANGLE
const vec4 vertexPositions[3] =
    vec4[](vec4(0.0f, 1.0f, 0.0f, 1.0f), vec4(-1.0f, -1.0f, 0.0f, 1.0f),
           vec4(1.0f, -1.0f, 0.0f, 1.0f));
#else
const vec4 vertexPositions[3] =
    vec4[](vec4(-1.0f, -1.0f, 0.0f, 1.0f), vec4(3.0f, -1.0f, 0.0f, 1.0f),
           vec4(-1.0f, 3.0f, 0.0f, 1.0f));
#endif

const vec3 vertexColors[3] = vec3[](
    vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f));

void main() {
  vec4 position = vertexPositions[gl_VertexIndex];
  v_color = vertexColors[gl_VertexIndex];
#ifdef MVP_TRIANGLE
  gl_Position = ubuf.mvp * position;
#else
  gl_Position = position;
#endif
}
