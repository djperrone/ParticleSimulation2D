#include "CudaMath.cuh"
#include "Novaura/CudaGLInterop/helper_cuda.h"

#define NUM_THREADS 256
#define NUM_PARTICLES 1000

namespace CudaMath {	

	void __global__ CudaMath::MakeTranslationMatrices_gpu(Matrix44f* matrices, common::particle_t* particles, size_t numParticles)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= numParticles) return;

		MAKE_TRANSLATION_xyz(matrices[tid], (float)particles[tid].x, (float)particles[tid].y, 0.0f);
	}

	void CudaMath::MakeTranslationMatrices_cpu(Matrix44f* matrices, common::particle_t* particles, size_t numParticles)
	{
		int num_blocks = (numParticles + NUM_THREADS - 1) / NUM_THREADS;

		MakeTranslationMatrices_gpu CUDA_KERNEL(num_blocks, NUM_THREADS)(matrices, particles, numParticles);
		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("translation kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
			exit(-1);
		}
	}	

	void CudaMath::MatMul44_cpu(Matrix44f* A, Matrix44f* B, Matrix44f* C, int N)
	{		
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				for (int k = 0; k < 4; k++)
				{
					C->rows[i].vec[j] += A->rows[k].vec[j] * B->rows[i].vec[k];
				}
			}
		}
	}
	__global__ void MatMul44Batch_gpu(Matrix44f* inGrid, Matrix44f* B, Matrix44f* outGrid, int numParticles)
	{	
		int tid = blockDim.x * blockIdx.x + threadIdx.x;

		/*	const int localSize = NUM_THREADS / 16;
			__shared__ Matrix44f localGrid[localSize];

			__shared__ Matrix44f localB;
			memcpy(&localB, B, sizeof(Matrix44f));*/

		if (tid >= numParticles * 16 || blockIdx.x >= numParticles) return;

		//memcpy(localGrid, inGrid + blockIdx.x * localSize, sizeof(Matrix44f) * localSize);

		int i = tid / 16;
		int j = tid % 16;

		int row = j / 4;
		int col = j % 4;

		float tmpSum = 0;
		//int local_i = i % localSize;
		//printf("blkidx: %i, i: %i, local_i: %i, local_size: %i\n",blockIdx.x,i ,local_i, localSize);

		//tmpSum += localGrid[local_i].rows[0].vec[col] * localB.rows[row].vec[0];
		//tmpSum += localGrid[local_i].rows[1].vec[col] * localB.rows[row].vec[1];
		//tmpSum += localGrid[local_i].rows[2].vec[col] * localB.rows[row].vec[2];
		//tmpSum += localGrid[local_i].rows[3].vec[col] * localB.rows[row].vec[3];

		tmpSum += inGrid[i].rows[0].vec[col] * B->rows[row].vec[0];
		tmpSum += inGrid[i].rows[1].vec[col] * B->rows[row].vec[1];
		tmpSum += inGrid[i].rows[2].vec[col] * B->rows[row].vec[2];
		tmpSum += inGrid[i].rows[3].vec[col] * B->rows[row].vec[3];

		//__syncthreads();
		outGrid[i].mat[j] = tmpSum;
	}

	//__global__ void MatMul44Batch_gpu(Matrix44f* inGrid, Matrix44f* B, Matrix44f* outGrid, int* numParticles)
	//{
	//	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	///*	const int localSize = NUM_THREADS / 16;
	//	__shared__ Matrix44f localGrid[localSize];

	//	__shared__ Matrix44f localB;
	//	memcpy(&localB, B, sizeof(Matrix44f));*/

	//	if (tid >= *numParticles * 16 || blockIdx.x >= *numParticles) return;

	//	//memcpy(localGrid, inGrid + blockIdx.x * localSize, sizeof(Matrix44f) * localSize);

	//	int i = tid / 16;
	//	int j = tid % 16;

	//	int row = j / 4;
	//	int col = j % 4;

	//	float tmpSum = 0;
	//	//int local_i = i % localSize;
	//	//printf("blkidx: %i, i: %i, local_i: %i, local_size: %i\n",blockIdx.x,i ,local_i, localSize);

	//	//tmpSum += localGrid[local_i].rows[0].vec[col] * localB.rows[row].vec[0];
	//	//tmpSum += localGrid[local_i].rows[1].vec[col] * localB.rows[row].vec[1];
	//	//tmpSum += localGrid[local_i].rows[2].vec[col] * localB.rows[row].vec[2];
	//	//tmpSum += localGrid[local_i].rows[3].vec[col] * localB.rows[row].vec[3];

	//		tmpSum += inGrid[i].rows[0].vec[col] * B->rows[row].vec[0];
	//		tmpSum += inGrid[i].rows[1].vec[col] * B->rows[row].vec[1];
	//		tmpSum += inGrid[i].rows[2].vec[col] * B->rows[row].vec[2];
	//		tmpSum += inGrid[i].rows[3].vec[col] * B->rows[row].vec[3];

	//		//__syncthreads();
	//	outGrid[i].mat[j] = tmpSum;		
	//}

	void MatMul44Batch_cpu(Matrix44f* inGrid, Matrix44f* B, Matrix44f* outGrid, int numParticles)
	{
		int num_blocks = (numParticles * 16 + NUM_THREADS - 1) / NUM_THREADS;
		//int* numParticles_d;
		//cudaMalloc((void**)&numParticles_d, sizeof(int));
		//cudaMemcpy(numParticles_d, &numParticles, sizeof(int), cudaMemcpyHostToDevice);

		MatMul44Batch_gpu CUDA_KERNEL(num_blocks, NUM_THREADS) (inGrid, B, outGrid, numParticles);
		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("matmul44batch kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
			exit(-1);
		}

		//cudaFree(numParticles_d);
	}

	__global__ void MatMul44_gpu(Matrix44f* A, Matrix44f* B, Matrix44f* C, int N)
	{
		int test = threadIdx.x;
		printf("\nasdasdan");

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < N; k++)
				{
					C->rows[i].vec[j] += A->rows[k].vec[j] * B->rows[i].vec[k];
				}
			}
		}
		printf("\n--fgdfgdfgd--\n");
	}		
}