#pragma once
#include "common/particle_t.h"

#define MAX_PARTICLES 25

namespace common {

    typedef struct Block {
        common::particle_t* particles[MAX_PARTICLES]; // filled in 0 to pcount-1
        int ids[MAX_PARTICLES]; // locations in original partical array
        int pcount = 0;
    }Block;

    void push_particle(Block& b, particle_t* particle, int idx);
    void erase_particle(Block& b, int idx);
}
