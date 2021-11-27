#pragma once
//#ifndef BLOCK_CUH
//#define BLOCK_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common/Block.h"

//#include "common/common.h"

namespace {   

   /* __host__ __device__ void push_particle_gpu(Block& b, common::particle_t* particle, int idx);
    __host__ __device__ void erase_particle_gpu(Block& b, int idx);*/
    __host__ __device__ void push_particle_gpu(common::Block& b, common::particle_t* particle, int idx) {      
        
        b.particles[b.pcount] = particle;
        b.ids[b.pcount] = idx;
        b.pcount++;
    }

    __host__ __device__ void erase_particle_gpu(common::Block& b, int idx) {

        while (idx < b.pcount - 1) {
            b.particles[idx] = b.particles[idx + 1];
            b.ids[idx] = b.ids[idx + 1];
            idx++;
        }

        b.pcount--;
    }
}
//#endif // !Block

