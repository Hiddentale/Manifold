#version 450
#extension GL_EXT_multiview : enable

struct MeshChunkInfo {
    vec3 aabb_min;
    uint voxel_slot;
    vec3 aabb_max;
    uint boundary_slot;
    ivec3 chunk_pos;
    uint face_id;
};

layout(set = 0, binding = 1) uniform CameraUBO {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
} camera;

layout(std430, set = 0, binding = 3) readonly buffer FacesBuffer {
    uvec2 faces[];
};

layout(std430, set = 0, binding = 4) readonly buffer ChunkInfoBuffer {
    MeshChunkInfo chunks[];
};

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragNormalWorld;
layout(location = 2) flat out uint fragMaterialId;
layout(location = 3) out vec3 fragWorldPos;
layout(location = 4) flat out ivec3 fragBlockCell;
layout(location = 5) flat out uint fragLocalFace;

const vec3 FACE_NORMALS[6] = vec3[6](
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
    vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1)
);

const vec3 FACE_TANGENTS[6] = vec3[6](
    vec3(0, 0, 1), vec3(0, 0, -1), vec3(1, 0, 0),
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(1, 0, 0)
);

const vec3 FACE_BITANGENTS[6] = vec3[6](
    vec3(0, 1, 0), vec3(0, 1, 0), vec3(0, 0, 1),
    vec3(0, 0, -1), vec3(0, 1, 0), vec3(0, 1, 0)
);

const vec2 CORNER_UVS[4] = vec2[4](
    vec2(0, 1), vec2(1, 1), vec2(1, 0), vec2(0, 0)
);

vec3 project_chunk_local(ivec3 cp, vec3 local) {
    return vec3(cp) * 16.0 + local;
}


const uint TRI_TO_CORNER[6] = uint[6](0u, 1u, 2u, 0u, 2u, 3u);

void main() {
    uint face_index = uint(gl_VertexIndex) / 6u;
    uint corner = TRI_TO_CORNER[uint(gl_VertexIndex) % 6u];

    uvec2 rec = faces[face_index];
    uint chunk_idx = rec.x;
    uint packed = rec.y;
    uint gx = packed & 0x1Fu;
    uint gy = (packed >> 5) & 0x1Fu;
    uint gz = (packed >> 10) & 0x1Fu;
    uint face = (packed >> 15) & 0x7u;
    uint material = (packed >> 18) & 0xFFu;

    MeshChunkInfo chunk = chunks[chunk_idx];
    vec3 center = vec3(float(gx) + 0.5, float(gy) + 0.5, float(gz) + 0.5);
    vec3 n = FACE_NORMALS[face];
    vec3 t = FACE_TANGENTS[face];
    vec3 b = FACE_BITANGENTS[face];
    vec3 face_center = center + n * 0.5;

    float u_off = (corner == 0u || corner == 3u) ? -0.5 : 0.5;
    float v_off = (corner < 2u) ? -0.5 : 0.5;
    vec3 local_pos = face_center + t * u_off + b * v_off;
    vec3 world_pos = project_chunk_local(chunk.chunk_pos, local_pos);

    gl_Position = camera.view_projection[gl_ViewIndex] * vec4(world_pos, 1.0);
    fragWorldPos = world_pos;
    fragTexCoord = CORNER_UVS[corner];
    fragNormalWorld = n;
    fragMaterialId = material;
    fragBlockCell = chunk.chunk_pos * 16 + ivec3(int(center.x), int(center.y), int(center.z));
    fragLocalFace = face;
}
