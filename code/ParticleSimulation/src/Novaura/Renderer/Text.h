#pragma once
#include <glm/glm.hpp>

namespace Novaura {

	class Text
	{
	public:



	public:
		struct Character {
			unsigned int TextureID;  // ID handle of the glyph texture
			glm::ivec2   Size;       // Size of glyph
			glm::ivec2   Bearing;    // Offset from baseline to left/top of glyph
			unsigned int Advance;    // Offset to advance to next glyph
		};

	private:

	};
}