#version 450

layout(location = 0) in vec3 worldPos;
layout(location = 0) out vec4 outColor;
layout(binding = 1) uniform sampler2D particleTex;

void main() {
    // 点精灵圆形裁剪
    vec2 coord = gl_PointCoord - vec2(0.5);
    float alpha = 0.5 - length(coord);
    if (alpha <= 0.0) discard;

    // 球体参数
    float radius = 1; // 与C++侧一致

    // 计算方向和距离
    vec3 dir = normalize(worldPos);
    float r = length(worldPos) / radius; // [0,1]

    // 计算UV：r=1时采样图片中心，r=0时采样图片边缘
    float u = 0.5 + dir.x * r * 0.5;
    float v = 0.5 - dir.y * r * 0.5;
    vec2 uv = vec2(u, v);

    vec4 texColor = texture(particleTex, uv);
    if(texColor.a < 0.1) discard;

    outColor = texColor;
}