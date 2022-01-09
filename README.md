# ParticleSimulation2D

![image](https://github.com/djperrone/ParticleSimulation2D/blob/main/sample_img/sample.JPG)

Rendering a particle simulation homework assignment from my parallel computing course.  This was mainly a large test-run for my stable fluids project as I tested using cuda, opengl-cuda interop, and instanced rendering.  This is more difficult to run than Stable Fluids because the cu and cuh files must be manually included and they are spread throughout several folders (something I addressed in Stable Fluids by consolidating them).  

### How To Run:
- git clone
- git submodule update --init --recursive
- Run the GenerateProjects batch file
- Open the project and manually include cu and cuh files in the project
- F5 to run (Release mode recommended)
