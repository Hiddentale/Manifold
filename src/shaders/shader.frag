#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragNormalWorld;
layout(location = 2) flat in uint fragMaterialId;
layout(location = 3) in vec3 fragWorldPos;
layout(location = 4) flat in ivec3 fragBlockCell;
layout(location = 5) flat in uint fragLocalFace;

layout(binding = 0) uniform sampler2DArray texArray;

layout(binding = 1) uniform UniformBufferObject {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
} ubo;

layout(location = 0) out vec4 outColor;

const float EDGE_WIDTH = 0.02;

// Hash a block-grid cell to a random value in [0, 1).
float hash1(vec2 cell) {
    return fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
}

// Stochastic tiling: randomly flips UV axes per block cell to break repetition.
vec3 stochastic_sample(uint layer, vec2 faceUV, vec2 worldCell, float atlasOffsetU) {
    float r = hash1(worldCell);
    bool flipX = r > 0.5;
    // Only flip Y for the top face — side faces have a directional gradient
    // (grass at top, dirt at bottom) that must not be inverted.
    bool flipY = atlasOffsetU > 0.25 && fract(r * 3.7) > 0.5;
    vec2 uv = vec2(flipX ? 1.0 - faceUV.x : faceUV.x,
                   flipY ? 1.0 - faceUV.y : faceUV.y);
    // Scale x into the correct atlas half [0, 0.5] or [0.5, 1.0]
    uv.x = uv.x * 0.5 + atlasOffsetU;
    return texture(texArray, vec3(uv, float(layer))).rgb;
}

void main() {
    // Stochastic-tiling cell is the block's stable integer
    // identity (chunk_pos*16 + block local), so the rotation is
    // constant per block regardless of how curved the projection
    // makes the face. The atlas top/side split is keyed off the
    // block's *local* face id, not the world normal — this works for
    // any cube face since "top" of a block on the planet means the
    // block's local +Y face direction.
    bool isTop = fragLocalFace == 2u;
    float atlasOffset = isTop ? 0.5 : 0.0;
    vec2 cell = vec2(float(fragBlockCell.x ^ (fragBlockCell.y * 73)),
                     float(fragBlockCell.z ^ (fragBlockCell.y * 19)));
    vec3 baseColor = stochastic_sample(fragMaterialId, fragTexCoord, cell, atlasOffset);

    vec3 N = normalize(fragNormalWorld);
    vec3 L = normalize(-ubo.light_direction);
    float diffuse = max(dot(N, L), 0.0);
    vec3 finalColor = (ubo.ambient_strength + diffuse) * baseColor;

    outColor = vec4(finalColor, 1.0);
}
