#include "../CudaSrc/Block.cuh"

namespace Physics {


  /*  __host__ __device__ void push_particle_gpu(Block& b, common::particle_t* particle, int idx) {
        b.particles[b.pcount] = particle;
        b.ids[b.pcount] = idx;
        b.pcount++;
    }

    __host__ __device__ void erase_particle_gpu(Block& b, int idx) {

        while (idx < b.pcount - 1) {
            b.particles[idx] = b.particles[idx + 1];
            b.ids[idx] = b.ids[idx + 1];
            idx++;
        }

        b.pcount--;
    }*/
}