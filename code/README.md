# SpaceAdventuresGL
Remaking SpaceAdventures in C++ using OpenGL to build a 2d renderer

Link to original game made in Java as part of a group project for school: https://github.com/djperrone/SpaceAdventures

#### New Features:
- Ship Rotation
- Target Icon
- Ammo/Reload system and Rate of Fire
- Pause Menu
- Play Again button

## Instructions to Play:
- Clone Repo
- Run GenerateProjects.batch
- Open generated Visual Studio Project and hit f5
- WASD to move, left mouse click to fire gun

#### Currently In Progress:<br>
The core part of the game runs and is playable but there are still more features I would like to add <br>

### Code Details
- The code in this project is split into two main folders (and namespaces).
- Novaura is the engine.  It contains rendering code and any opengl code that is used to create an application.
- SpaceAdventures contains the code specifically used for gameplay.
- The idea here is to make it easy to re-use this project to create a different game in the future by separating the gameplay from the engine.
- The program starts from the main.cpp file found in the Main folder.  It launches an application that inherits from Novaura::Application
- A state machine is used to control the flow of the application. 

### TODO:
- Text Rendering
- Particle System
- Lighting Affects




The opengl code that I'm writing here is based on code from the learnopengl series and the cherno's opengl and Hazel series.

https://learnopengl.com/ <br>
https://www.youtube.com/user/TheChernoProject
