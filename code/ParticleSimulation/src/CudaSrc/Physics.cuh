#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common/common.h"
#include "Block.cuh"


namespace Physics {

	__device__ void apply_force_gpu(common::particle_t& particle, common::particle_t& neighbor);
	__device__ void apply_within_block_gpu(Block& block);
	__device__ void apply_across_blocks_gpu(Block& block1, Block& block2);
	__global__ void compute_forces_gpu(Block* grid, int blocks_per_side);
	__global__ void move_gpu(Block* grid, int blocks_per_side, double size);
	__global__ void check_move_gpu(Block* grid, int blocks_per_side, double block_size);

}