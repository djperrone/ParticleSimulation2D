//#include "sapch.h"

#include "physics.cuh"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common/common_gpu.cuh"
#include "common/common.h"
#include "../common/ParticleData.h"
#include "common/Block.h"
#include "BlockFunctions.cuh"
#include "common/PVec.h"

// https://stackoverflow.com/questions/6061565/setting-up-visual-studio-intellisense-for-cuda-kernel-calls
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

namespace Physics {

    __device__ void apply_force_gpu(common::particle_t& particle, common::particle_t& neighbor)
    {
        double dx = neighbor.x - particle.x;
        double dy = neighbor.y - particle.y;
        double r2 = dx * dx + dy * dy;
        if (r2 > cutoff1 *cutoff1)
            return;
        //r2 = fmax( r2, min_r*min_r );
        r2 = (r2 > min_r1 * min_r1) ? r2 : min_r1 * min_r1;
        double r = sqrt(r2);

        //
        //  very simple short-range repulsive force
        //
        double coef = (1 - cutoff1 / r) / r2 / mass1;
        particle.ax += coef * dx;
        particle.ay += coef * dy;
    }

    __device__ void apply_within_block_gpu(common::Block& block) {
        for (int i = 0; i < block.pcount; i++) {
            for (int j = 0; j < block.pcount; j++) {
                apply_force_gpu(*block.particles[i], *block.particles[j]);
            }
        }
    }

    __device__ void apply_across_blocks_gpu(common::Block& block1, common::Block& block2) {
        for (int i = 0; i < block1.pcount; i++) {
            for (int j = 0; j < block2.pcount; j++) {
                apply_force_gpu(*block1.particles[i], *block2.particles[j]);
            }
        }
    }

    __global__ void compute_forces_gpu(common::Block* grid, int blocks_per_side)
    {

        //int tid = threadIdx.x + blockIdx.x * blockDim.x;
        //if (tid >= blocks_per_side * blocks_per_side) return;

        //int i = tid / blocks_per_side;
        //int j = tid % blocks_per_side;

        //common::Block* curr = &grid[i * blocks_per_side + j];
        ////set acc to 0
        //for (int k = 0; k < curr->pcount; k++) {
        //    curr->particles[k]->ax = curr->particles[k]->ay = 0;
        //}
       // printf(__FUNCTION__);

        // Get thread (particle) ID
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= blocks_per_side * blocks_per_side) return;

        int i = tid / blocks_per_side;
        int j = tid % blocks_per_side;

        //if (i * blocks_per_side + j >= blocks_per_side * blocks_per_side) return;

        common::Block& curr = grid[i * blocks_per_side + j];
       /* printf("blocks_per_side: %i\n", blocks_per_side );
        printf("i: %i, j: %i\n", i, j);
        printf("index: %i\n", i * blocks_per_side + j);
        printf("tid: %i\n", tid);*/
        //printf("pcount: %i\n",  curr.pcount);

        //set acc to 0
        for (int k = 0; k < curr.pcount; k++) {
            printf("tid:%i, x: %f, y: %f\n", tid, curr.particles[k]->x, curr.particles[k]->y);
           // printf("test");
            curr.particles[k]->ax = curr.particles[k]->ay = 0;
        }
       // printf("success!\n");
        // check each of 8 neighbors exists, calling apply_across_blocks_gpu if so
        if (j != blocks_per_side - 1) { //right
            apply_across_blocks_gpu(curr, grid[i * blocks_per_side + j + 1]);
           // printf("test1");

        }
        if (j != blocks_per_side - 1 && i != blocks_per_side - 1) { //down+right
            apply_across_blocks_gpu(curr, grid[(i + 1) * blocks_per_side + j + 1]);
           // printf("test2");

        }
        if (j != blocks_per_side - 1 && i != 0) { //up+right
            apply_across_blocks_gpu(curr, grid[(i - 1) * blocks_per_side + j + 1]);
        }
        if (i != 0) { //up
            apply_across_blocks_gpu(curr, grid[(i - 1) * blocks_per_side + j]);
        }
        if (i != blocks_per_side - 1) { //down
            apply_across_blocks_gpu(curr, grid[(i + 1) * blocks_per_side + j]);
        }
        if (j != 0) { //left
            apply_across_blocks_gpu(curr, grid[i * blocks_per_side + j - 1]);
        }
        if (j != 0 && i != 0) { //up+left
            apply_across_blocks_gpu(curr, grid[(i - 1) * blocks_per_side + j - 1]);
        }
        if (j != 0 && i != blocks_per_side - 1) { //down+left
            apply_across_blocks_gpu(curr, grid[(i + 1) * blocks_per_side + j - 1]);
        }

        // apply forces within the block
        apply_within_block_gpu(curr);
    }

    __global__ void move_gpu(common::Block* grid, int blocks_per_side, double size)
    {
        //printf(__FUNCTION__);

        // Get thread (particle) ID
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= blocks_per_side * blocks_per_side) return;

        int i = tid / blocks_per_side;
        int j = tid % blocks_per_side;

        common::Block& curr = grid[i * blocks_per_side + j];
        //
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        //
        for (int k = 0; k < curr.pcount; k++) {
            common::particle_t* p = curr.particles[k];
            p->vx += p->ax * dt1;
            p->vy += p->ay *dt1;
            p->x += p->vx * dt1;
            p->y += p->vy * dt1;

            //
            //  bounce from walls
            //
            while (p->x < 0 || p->x > size)
            {
                p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
                p->vx = -(p->vx);
            }
            while (p->y < 0 || p->y > size)
            {
                p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
                p->vy = -(p->vy);
            }
        }
    }

    __global__ void check_move_gpu(common::Block* grid, int blocks_per_side, double block_size)
    {
        //printf(__FUNCTION__);

        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= blocks_per_side * blocks_per_side) return;

        int old_block_x = tid / blocks_per_side;
        int old_block_y = tid % blocks_per_side;

        common::Block& curr = grid[old_block_x * blocks_per_side + old_block_y];

        for (int k = 0; k < curr.pcount; k++) {
            int new_block_x = (int)(curr.particles[k]->x / block_size);
            int new_block_y = (int)(curr.particles[k]->y / block_size);

            // if particle moved to a new block, remove from old and put in new
            if (old_block_x != new_block_x || old_block_y != new_block_y) {
                push_particle_gpu(grid[new_block_x * blocks_per_side + new_block_y], curr.particles[k], curr.ids[k]);
                erase_particle_gpu(curr, k);
            }
        }

    }

    __global__ void InitGrid_gpu(common::Block* grid, common::particle_t* particles, int blocks_per_side, double block_size, int n)
    {
        printf("init_grid_gpu\n");
        for (int i = 0; i < blocks_per_side * blocks_per_side; i++) {
            grid[i].pcount = 0;
        }

        for (size_t i = 0; i < n; i++) {
            int block_x = (int)(particles[i].x / block_size);
            int block_y = (int)(particles[i].y / block_size);
            push_particle_gpu(grid[block_x * blocks_per_side + block_y], &particles[i], i);
        }
    }

    void InitGrid(common::Block* grid, common::particle_t* particles, int blocks_per_side, double block_size, int n)
    {
        printf("init_grid\n");

        InitGrid_gpu CUDA_KERNEL(1, 1) (grid, particles, blocks_per_side, block_size, n);
    }

    __global__ void Test()
    {
        printf("Test function\n");
    }

    void compute_forces(int blks, int numThreads, common::Block* grid, int blocks_per_side)
    {
       // printf(__FUNCTION__);
      

        compute_forces_gpu CUDA_KERNEL(blks, NUM_THREADS) (grid, blocks_per_side);
        cudaError_t cudaerr = cudaDeviceSynchronize();

        if (cudaerr != cudaSuccess)
        {
            printf("2 kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
            __debugbreak;
        }

    }

    void move(int blks, int numThreads, common::Block* grid, int blocks_per_side, double size)
    {
        //printf(__FUNCTION__);

        move_gpu CUDA_KERNEL(blks, numThreads)(grid, blocks_per_side, size);
    }

    void check_move(int blks, int numThreads, common::Block* grid, int blocks_per_side, double block_size)
    {
        //printf(__FUNCTION__);
        //printf(__FUNCTION__);

        check_move_gpu CUDA_KERNEL(blks, numThreads) (grid, blocks_per_side, block_size);
    }

    void InitParticles(common::particle_t* particles, common::particle_t* d_particles)
    {
        cudaThreadSynchronize();
        particles = (common::particle_t*)malloc(common::ParticleData::num_particles * sizeof(common::particle_t));
        common::set_size(common::ParticleData::num_particles);

        common::init_particles(common::ParticleData::num_particles, particles);
       // common_gpu::ParticleDataGPU deviceParticleData;
        
       // deviceParticleData.Init();

        cudaMalloc((void**)&d_particles, common::ParticleData::num_particles * sizeof(common::particle_t));

       

    }
    void ShutDown()
    {
    }



    void apply_within_block (pvec::ParticleVec particles){
        int num_particles = particles.pcount;
        for(int i = 0; i < num_particles; i++){
            for(int j = 0; j < num_particles; j++){
                common::apply_force(*particles.data[i], *particles.data[j]);
            }
        }
    }

    void apply_across_blocks (pvec::ParticleVec particles1, pvec::ParticleVec particles2){
        int num_particles1 = particles1.pcount;
        int num_particles2 = particles2.pcount;
        for(int i = 0; i < num_particles1; i++){
            for(int j = 0; j < num_particles2; j++){
                common::apply_force(*particles1.data[i], *particles2.data[j]);
            }
        }
    }

    void check_move(pvec::ParticleVec** grid, common::particle_t* particle, double old_x, double old_y, double block_size) {
        int old_block_x = (int) old_x / block_size;
        int old_block_y = (int) old_y / block_size;

        int new_block_x = (int) particle->x / block_size;
        int new_block_y = (int) particle->y / block_size;

        // if particle moved to a new block, remove from old and put in new
        if(old_block_x != new_block_x || old_block_y != new_block_y){
            for(int i = 0; i < grid[old_block_x][old_block_y].pcount; i++){
                if(grid[old_block_x][old_block_y].data[i]->x == particle->x && grid[old_block_x][old_block_y].data[i]->y == particle->y){
                    pvec::EraseParticle(grid[old_block_x][old_block_y], i);
                    pvec::PushParticle(grid[new_block_x][new_block_y], particle);
                    break;
                }
            }
        }
    }
    //void check_move(pvec::ParticleVec** grid, common::particle_t* particle, double old_x, double old_y, double block_size)
    //{
    //    int old_block_x = (int)old_x / block_size;
    //    int old_block_y = (int)old_y / block_size;

    //    int new_block_x = (int)particle->x / block_size;
    //    int new_block_y = (int)particle->y / block_size;

    //    // if particle moved to a new block, remove from old and put in new
    //    if (old_block_x != new_block_x || old_block_y != new_block_y) {
    //        for (int i = 0; i < grid[old_block_x][old_block_y].pcount; i++) {
    //            if (grid[old_block_x][old_block_y].data[i]->x == particle->x && grid[old_block_x][old_block_y].data[i]->y == particle->y) {
    //                pvec::EraseParticle(grid[old_block_x][old_block_y], i);
    //                pvec::PushParticle(grid[new_block_x][new_block_y], particle);
    //                break;
    //            }
    //        }
    //    }
    //}
}
