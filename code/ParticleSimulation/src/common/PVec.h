#pragma once
#include "common.h"

namespace pvec {


    typedef struct {
        int capacity;
        int pcount;
       common::particle_t** data;
    }ParticleVec;

    // add and remove particles from vector
    void PushParticle(ParticleVec& vec, common::particle_t* p);
    void EraseParticle(ParticleVec& vec, int index);

    // initialize 3D grid of particles
    ParticleVec** InitGrid3d(int length);
    void FreeGrid(ParticleVec** grid, int length);

    // print grid for testing purposes
    void PrintVec(ParticleVec** grid, int blocks_per_side);


}
