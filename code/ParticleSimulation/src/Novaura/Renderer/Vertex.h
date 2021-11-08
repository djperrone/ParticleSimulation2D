#pragma once

#include <glm/glm.hpp>

namespace Novaura{
	struct VertexData
	{
		VertexData(const::glm::vec3& position, const glm::vec4& color, const glm::vec2& texCoord, const glm::vec2& quantity, float slot)
			: Position(position), Color(color), TexCoord(texCoord), Quantity(quantity), TextureSlot(slot){}
		VertexData(const::glm::vec3& position, const glm::vec4& color, const glm::vec2& texCoord, const glm::vec2& quantity) : Position(position), Color(color), TexCoord(texCoord), Quantity(quantity) {}
		VertexData(const::glm::vec3& position, const glm::vec4& color, const glm::vec2& texCoord) : Position(position), Color(color), TexCoord(texCoord) {}
		glm::vec3 Position;
		glm::vec4 Color;
		glm::vec2 TexCoord;
		glm::vec2 Quantity = glm::vec2(1.0f, 1.0f);
		float TextureSlot = 0.0f;
	};
}

