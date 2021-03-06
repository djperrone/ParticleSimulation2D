#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../common/particle_t.h"
#include "common/PVec.h"
#include "common/Block.h"
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>

#define NUM_THREADS 256

struct Block;

namespace Physics {

	__device__ void apply_force_gpu(common::particle_t& particle, common::particle_t& neighbor);
	__device__ void apply_within_block_gpu(common::Block& block);
	__device__ void apply_across_blocks_gpu(common::Block& block1, common::Block& block2);
	__global__ void compute_forces_gpu(common::Block* grid, int blocks_per_side);
	__global__ void move_gpu(common::Block* grid, int blocks_per_side, double size);
	__global__ void check_move_gpu(common::Block* grid, int blocks_per_side, double block_size);
	__global__ void InitGrid_gpu(common::Block* grid, common::particle_t* particles, int blocks_per_side, double block_size, int n);
	__global__ void UpdateMatrices_gpu_glm(glm::mat4* matrices, common::particle_t* particles, size_t numParticles);

	//__global__ void check_move_serial_gpu(common::Block* grid, size_t numParticles);
	//void check_move_serial(common::Block* grid, size_t numParticles, double block_size);
	void __global__ check_move_serial_gpu(common::Block* grid, common::particle_t* particles, size_t num_particles, size_t blocks_per_side, double block_size);
	void check_move_serial(common::Block* grid, common::particle_t* particles_gpu, size_t num_particles, size_t blocks_per_side, double block_size);
	
	void UpdateMatrices_cpu_glm(glm::mat4* matrices, common::particle_t* particles, size_t numParticles);


	void InitGrid(common::Block* grid, common::particle_t* particles,int blocks_per_side, double block_size, int n);
	
	void compute_forces(int blks, int numThreads, common::Block* grid, int blocks_per_side);
	void move(int blks, int numThreads, common::Block* grid, int blocks_per_side, double size);
	void check_move_wrapper(int blks, int numThreads, common::Block* grid, int blocks_per_side, double block_size);

	void apply_within_block(pvec::ParticleVec particles);
	void apply_across_blocks(pvec::ParticleVec particles1, pvec::ParticleVec particles2);
	void check_move(common::Block& grid, common::particle_t* particle, double old_x, double old_y, double block_size);
	void check_move(pvec::ParticleVec** grid, common::particle_t* particle, double old_x, double old_y, double block_size);
	//void check_move(pvec::ParticleVec** grid, common::particle_t* particle, double old_x, double old_y, double block_size);
	void grid_particle_sim(common::particle_t* particles, int n, const double size, FILE* fsave);	

	void InitParticles(common::particle_t* particles, common::particle_t* d_particles);
	void InitGrid();
	void ShutDown();

}

#endif