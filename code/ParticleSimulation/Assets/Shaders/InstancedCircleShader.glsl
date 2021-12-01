#shader vertex

#version 330 core
layout(location = 0) in vec3 aPos;
//layout(location = 3) in vec2 aQuantity;

//layout(location = 1) in vec3 aNormal;
//layout(location = 2) in vec2 aTexCoords;
layout(location = 1) in mat4 instanceMatrix;

//out vec2 v_TexCoords;
out vec3 WorldPos;
//out vec3 Normal;

//uniform mat4 projection;
//uniform mat4 view;

uniform mat4 u_ViewProjectionMatrix;


void main()
{
    //v_TexCoords = aTexCoords;
    WorldPos = vec3(instanceMatrix * vec4(aPos, 1.0));
    //Normal = mat3(instanceMatrix) * aNormal;

    //gl_Position = projection * view * vec4(WorldPos, 1.0);
    gl_Position = u_ViewProjectionMatrix * instanceMatrix * vec4(aPos, 1.0);
}

#shader fragment
#version 330 core

out vec4 Color;

in vec3 WorldPos;

//in vec3 v_Pos;
in vec4 v_Color;
in vec4 v_TexCoords;
in vec2 v_Quantity;
//in float v_TexIndex;

//uniform sampler2D u_Textures[32];

void main()
{
   // float distance = 1.0 - length(vec2(WorldPos.x - v_TexCoords.x, WorldPos.y - v_TexCoords.y));
    //float distance = 1.0 - length(vec2(v_Pos.x, v_Pos.y));   
    //float fade = 0.005;
    //float cutoff = 1.0 - (v_Quantity.x / 2.0);
    //distance = smoothstep(cutoff, cutoff + fade, distance);

    Color = vec4(0.7f,0.2f,0.2f,1.0f);
}