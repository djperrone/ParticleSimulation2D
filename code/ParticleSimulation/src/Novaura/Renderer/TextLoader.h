#pragma once
#include "Novaura/Renderer/Text.h"

namespace Novaura {

    class TextLoader
    {
    public:
        static std::map<char, Text::Character> LoadedCharacters;
        static Text::Character LoadFont(char key);

        static void Init();
    private:

    };
}