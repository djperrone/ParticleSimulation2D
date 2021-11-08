#include "sapch.h"
#include "TextLoader.h"

#include <glad/glad.h>
#include <ft2build.h>
#include FT_FREETYPE_H  
#include <spdlog/spdlog.h>

namespace Novaura {

    // text loader
    std::map<char, Text::Character> TextLoader::LoadedCharacters;

    Text::Character TextLoader::LoadFont(char key)
    {
        auto text = LoadedCharacters.find(key);
        if (text != LoadedCharacters.end())
        {
            return text->second;
        }
        else
        {
            spdlog::error("Could not find Text glyph");
        }
        /*else
        {
            LoadedCharacters.insert(std::pair<char, Text::Character>(key, Text::Character(key)));
        }*/

    }

    void TextLoader::Init()
    {

        FT_Library ft;
        if (FT_Init_FreeType(&ft))
        {
            spdlog::error("ERROR::FREETYPE: Could not init FreeType Library");
            exit(1);
        }
        FT_Face face;
        if (FT_New_Face(ft, "Assets/Fonts/arial.ttf", 0, &face))
        {
            spdlog::error("ERROR::FREETYPE: Failed to load font");
            exit(1);
        }
        FT_Set_Pixel_Sizes(face, 0, 48);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // disable byte-alignment restriction

        for (unsigned char c = 0; c < 128; c++)
        {
            // load character glyph 
            if (FT_Load_Char(face, c, FT_LOAD_RENDER))
            {
                std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
                continue;
            }
            // generate texture
            unsigned int texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                face->glyph->bitmap.width,
                face->glyph->bitmap.rows,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                face->glyph->bitmap.buffer
            );
            // set texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            // now store character for later use
            Text::Character character = {
                texture,
                glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                face->glyph->advance.x
            };
            TextLoader::LoadedCharacters.insert(std::pair<char, Text::Character>(c, character));
        }

        FT_Done_Face(face);
        FT_Done_FreeType(ft);
    }

}