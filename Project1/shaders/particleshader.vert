#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
} mvp;

void main() {
    gl_PointSize = 14.0;
    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(inPosition, 1.0);

    // 计算世界空间距离
    float dist = length((mvp.model * vec4(inPosition, 1.0)).xyz - mvp.cameraPos);

    // 归一化距离（你可以根据实际场景调整maxDist）
    float maxDist = 2.0; // 你场景的最大可见距离
    float t = clamp(dist / maxDist, 0.0, 1.0);

    vec3 nearColor = vec3(1.0, 1.0, 0.0); // 亮黄
    vec3 farColor  = vec3(0.0, 0.2, 1.0); // 深蓝
    fragColor = mix(nearColor, farColor, t);
}