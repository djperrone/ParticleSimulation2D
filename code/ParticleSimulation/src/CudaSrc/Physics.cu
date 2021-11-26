//#include "sapch.h"

#include "physics.cuh"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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

    __device__ void apply_within_block(Block& block) {
        for (int i = 0; i < block.pcount; i++) {
            for (int j = 0; j < block.pcount; j++) {
                apply_force_gpu(*block.particles[i], *block.particles[j]);
            }
        }
    }

    __device__ void apply_across_blocks(Block& block1, Block& block2) {
        for (int i = 0; i < block1.pcount; i++) {
            for (int j = 0; j < block2.pcount; j++) {
                apply_force_gpu(*block1.particles[i], *block2.particles[j]);
            }
        }
    }

    __global__ void compute_forces_gpu(Block* grid, int blocks_per_side)
    {
        // Get thread (particle) ID
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= blocks_per_side * blocks_per_side) return;

        int i = tid / blocks_per_side;
        int j = tid % blocks_per_side;

        Block& curr = grid[i * blocks_per_side + j];
        //set acc to 0
        for (int k = 0; k < curr.pcount; k++) {
            curr.particles[k]->ax = curr.particles[k]->ay = 0;
        }

        // check each of 8 neighbors exists, calling apply_across_blocks if so
        if (j != blocks_per_side - 1) { //right
            apply_across_blocks(curr, grid[i * blocks_per_side + j + 1]);
        }
        if (j != blocks_per_side - 1 && i != blocks_per_side - 1) { //down+right
            apply_across_blocks(curr, grid[(i + 1) * blocks_per_side + j + 1]);
        }
        if (j != blocks_per_side - 1 && i != 0) { //up+right
            apply_across_blocks(curr, grid[(i - 1) * blocks_per_side + j + 1]);
        }
        if (i != 0) { //up
            apply_across_blocks(curr, grid[(i - 1) * blocks_per_side + j]);
        }
        if (i != blocks_per_side - 1) { //down
            apply_across_blocks(curr, grid[(i + 1) * blocks_per_side + j]);
        }
        if (j != 0) { //left
            apply_across_blocks(curr, grid[i * blocks_per_side + j - 1]);
        }
        if (j != 0 && i != 0) { //up+left
            apply_across_blocks(curr, grid[(i - 1) * blocks_per_side + j - 1]);
        }
        if (j != 0 && i != blocks_per_side - 1) { //down+left
            apply_across_blocks(curr, grid[(i + 1) * blocks_per_side + j - 1]);
        }

        // apply forces within the block
        apply_within_block(curr);
    }

    __global__ void move_gpu(Block* grid, int blocks_per_side, double size)
    {

        // Get thread (particle) ID
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= blocks_per_side * blocks_per_side) return;

        int i = tid / blocks_per_side;
        int j = tid % blocks_per_side;

        Block& curr = grid[i * blocks_per_side + j];
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

    __global__ void check_move_gpu(Block* grid, int blocks_per_side, double block_size)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= blocks_per_side * blocks_per_side) return;

        int old_block_x = tid / blocks_per_side;
        int old_block_y = tid % blocks_per_side;

        Block& curr = grid[old_block_x * blocks_per_side + old_block_y];

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
}