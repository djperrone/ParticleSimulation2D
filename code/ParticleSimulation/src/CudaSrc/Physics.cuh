#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../common/particle_t.h"

#define NUM_THREADS 256

struct Block;

namespace Physics {

	__device__ void apply_force_gpu(common::particle_t& particle, common::particle_t& neighbor);
	__device__ void apply_within_block_gpu(Block& block);
	__device__ void apply_across_blocks_gpu(Block& block1, Block& block2);
	__global__ void compute_forces_gpu(Block* grid, int blocks_per_side);
	__global__ void move_gpu(Block* grid, int blocks_per_side, double size);
	__global__ void check_move_gpu(Block* grid, int blocks_per_side, double block_size);

	/*void apply_force(common::particle_t& particle, common::particle_t& neighbor);
	void apply_within_block(Block& block);
	void apply_across_blocks(Block& block1, Block& block2);
	void compute_forces(Block* grid, int blocks_per_side);
	void move(Block* grid, int blocks_per_side, double size);
	void check_move(Block* grid, int blocks_per_side, double block_size);*/

	//void apply_force_wrapper(common::particle_t& particle, common::particle_t& neighbor);

	void InitParticles(common::particle_t* particles, common::particle_t* d_particles);
	void InitGrid();
	void ShutDown();

	

}

#endif