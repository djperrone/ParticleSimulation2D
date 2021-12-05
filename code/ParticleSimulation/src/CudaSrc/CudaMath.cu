#include "CudaMath.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <time.h>

#include "Novaura/CudaGLInterop/helper_cuda.h"


#define NUM_THREADS 256
#define NUM_PARTICLES 1000
//#define NUM_BLOCKS (NUM_PARTICLES * 16 + NUM_THREADS - 1) / NUM_THREADS)

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
	__global__ void MatMul44Batch_gpu(Matrix44f* grid, Matrix44f* B, Matrix44f* C, int numParticles)
	{	
		int tid = blockDim.x * blockIdx.x + threadIdx.x;		
		
		const int localSize = NUM_THREADS / 16;
		__shared__ Matrix44f localGrid[localSize];		

		__shared__ Matrix44f localB;
		memcpy(&localB, B, sizeof(Matrix44f));

		if (tid >= NUM_PARTICLES * 16 || blockIdx.x >= NUM_PARTICLES) return;

		memcpy(localGrid, grid + blockIdx.x * localSize, sizeof(Matrix44f) * localSize);

		int i = tid / 16;
		int j = tid % 16;

		int row = j / 4;
		int col = j % 4;		
		
		float tmpSum = 0;
		int local_i = i % localSize;
		//printf("blkidx: %i, i: %i, local_i: %i, local_size: %i\n",blockIdx.x,i ,local_i, localSize);

		tmpSum += localGrid[local_i].rows[0].vec[col] * localB.rows[row].vec[0];
		tmpSum += localGrid[local_i].rows[1].vec[col] * localB.rows[row].vec[1];
		tmpSum += localGrid[local_i].rows[2].vec[col] * localB.rows[row].vec[2];
		tmpSum += localGrid[local_i].rows[3].vec[col] * localB.rows[row].vec[3];

	/*	tmpSum += grid[i].rows[0].vec[col] * localB.rows[row].vec[0];
		tmpSum += grid[i].rows[1].vec[col] * localB.rows[row].vec[1];
		tmpSum += grid[i].rows[2].vec[col] * localB.rows[row].vec[2];
		tmpSum += grid[i].rows[3].vec[col] * localB.rows[row].vec[3];*/
		
		//__syncthreads();
		C[i].mat[j] = tmpSum;		

		/*for (int k = 0; k < 4; k++)
		{
			tmpSum += grid[i].rows[k].vec[col] * B->rows[row].vec[k];
		}*/
	}

	__global__ void MatMul44Batch_gpu(Matrix44f* inGrid, Matrix44f* B, Matrix44f* outGrid, int* numParticles)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;

	/*	const int localSize = NUM_THREADS / 16;
		__shared__ Matrix44f localGrid[localSize];

		__shared__ Matrix44f localB;
		memcpy(&localB, B, sizeof(Matrix44f));*/

		if (tid >= *numParticles * 16 || blockIdx.x >= *numParticles) return;

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

		/*for (int k = 0; k < 4; k++)
		{
			tmpSum += grid[i].rows[k].vec[col] * B->rows[row].vec[k];
		}*/
	}

	void MatMul44Batch_cpu(Matrix44f* inGrid, Matrix44f* B, Matrix44f* outGrid, int numParticles)
	{
		int num_blocks = (numParticles * 16 + NUM_THREADS - 1) / NUM_THREADS;
		int* numParticles_d;
		cudaMalloc((void**)&numParticles_d, sizeof(int));
		cudaMemcpy(numParticles_d, &numParticles, sizeof(int), cudaMemcpyHostToDevice);

		MatMul44Batch_gpu CUDA_KERNEL(num_blocks, NUM_THREADS) (inGrid, B,outGrid, numParticles_d);
		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("matmul44batch kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
			exit(-1);
		}

		cudaFree(numParticles_d);
	}

	
	void MatMul44BatchTest_cpu()
	{
		for (int num_tests = 0; num_tests < 50; num_tests++)
		{
			srand((unsigned int)time(NULL));

			Matrix44f A[NUM_PARTICLES], B, C[NUM_PARTICLES];
			Matrix44f* A_d, * B_d, * C_d;


			glm::mat4 ref_mats[NUM_PARTICLES];

			float scale = (float)rand() / (float)(RAND_MAX / 55.0f);
			MAKE_SCALE(B, scale);

			glm::mat4 scale_ref = glm::scale(glm::mat4(1.0f), glm::vec3(scale));			

			for (int i = 0; i < NUM_PARTICLES; i++)
			{
				float x = (float)rand() / (float)(RAND_MAX / 55.0f);
				float y = (float)rand() / (float)(RAND_MAX / 55.0f);
				float z = (float)rand() / (float)(RAND_MAX / 55.0f);

				Vector3f transVec{ x, y, z };
				MAKE_TRANSLATION(A[i], transVec);

				ref_mats[i] = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
			}
			printf("test {%i}", __LINE__);

			for (int i = 0; i < NUM_PARTICLES; i++)
				ZERO_FLAT_MATRIX(C[i]);

			for (int k = 0; k < NUM_PARTICLES; k++)
				for (int i = 0; i < 4; i++)
					for (int j = 0; j < 4; j++)
					{
						if (A[k].rows[i].vec[j] != ref_mats[k][i][j])
						{
							printf("initial matrices dont match\n");
							return;
						}
					}
			printf("success\n");
			cudaError_t cudaerr = cudaDeviceSynchronize();
			if (cudaerr != cudaSuccess)
			{
				printf("1 kernel launch failed with error \"%s\".\n",
					cudaGetErrorString(cudaerr));
				__debugbreak;
			}

			cudaMalloc((void**)&B_d, sizeof(Matrix44f));
			cudaMalloc((void**)&A_d, sizeof(Matrix44f) * NUM_PARTICLES);
			cudaMalloc((void**)&C_d, sizeof(Matrix44f) * NUM_PARTICLES);

			cudaerr = cudaDeviceSynchronize();
			if (cudaerr != cudaSuccess)
			{
				printf("2 kernel launch failed with error \"%s\".\n",
					cudaGetErrorString(cudaerr));
				__debugbreak;
			}

			cudaMemcpy(B_d, &B, sizeof(Matrix44f), cudaMemcpyHostToDevice);
			cudaMemcpy(A_d, A, sizeof(Matrix44f) * NUM_PARTICLES, cudaMemcpyHostToDevice);
			cudaMemcpy(C_d, C, sizeof(Matrix44f) * NUM_PARTICLES, cudaMemcpyHostToDevice);


			int num_blocks = (NUM_PARTICLES * 16 + NUM_THREADS - 1) / NUM_THREADS;

			MatMul44Batch_gpu CUDA_KERNEL(num_blocks, NUM_THREADS) (A_d, B_d, C_d, NUM_PARTICLES);
			cudaerr = cudaDeviceSynchronize();
			if (cudaerr != cudaSuccess)
			{
				printf("4 kernel launch failed with error \"%s\".\n",
					cudaGetErrorString(cudaerr));
				exit(-1);
			}

			cudaMemcpy(A, A_d, sizeof(Matrix44f) * NUM_PARTICLES, cudaMemcpyDeviceToHost);
			cudaMemcpy(C, C_d, sizeof(Matrix44f) * NUM_PARTICLES, cudaMemcpyDeviceToHost);


			for (int k = 0; k < NUM_PARTICLES; k++)
				for (int i = 0; i < 4; i++)
					for (int j = 0; j < 4; j++)
					{
						//printf("%f, ", A[k].rows[i].vec[j]);
						if (A[k].rows[i].vec[j] != ref_mats[k][i][j])
						{
							printf("after matrices dont match\n");
							exit(-1);
						}
					}
			printf("success3\n");


			glm::mat4 results[NUM_PARTICLES];
			for (int i = 0; i < NUM_PARTICLES; i++)
			{
				results[i] = ref_mats[i] * scale_ref;
			}

			for (int k = 0; k < NUM_PARTICLES; k++)
				for (int i = 0; i < 4; i++)
					for (int j = 0; j < 4; j++)
					{
						//printf("%f, ", A[k].rows[i].vec[j]);
						if (C[k].rows[i].vec[j] != results[k][i][j])
						{
							printf("after matrices dont match\n");
							exit(-1);
						}

					}
			printf("success!!!!!!\n");
		}		
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

	void CudaMath::MatMul44Test_cpu2()
	{
		for (int i = 0; i < 50; i++)
		{
			srand((unsigned int)time(NULL));


			Matrix44f A, B, C;
			ZERO_FLAT_MATRIX(C);
			Matrix44f transMat;
			float x = (float)rand() / (float)(RAND_MAX / 55.0f);
			float y = (float)rand() / (float)(RAND_MAX / 55.0f);
			float z = (float)rand() / (float)(RAND_MAX / 55.0f);



			float scale = (float)rand() / (float)(RAND_MAX / 55.0f);


			MAKE_SCALE(B, scale);

			glm::mat4 ref_scale = glm::scale(glm::mat4(1.0f), glm::vec3(scale, scale, scale));

			Vector3f transVec{ x, y, z };
			MAKE_TRANSLATION(A, transVec);


			glm::mat4 ref_trans = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
			//Matrix44f grid[4];




			/*MatMul44Batch_gpu CUDA_KERNEL(1, 1) (&A, &transMat, &C, 4);
			cudaDeviceSynchronize();*/

			MatMul44_cpu(&A, &B, &C, 4);
			//cudaDeviceSynchronize();

			glm::mat4 ref_result = glm::scale(ref_trans, glm::vec3(scale,scale,scale));
			
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{

					printf("%f, ", C.rows[i].vec[j]);

				}
				printf("\n");
			}

			printf("\n----\n");
			for (int i = 0; i < 4; i++)\
			{
				for (int j = 0; j < 4; j++)
				{
					printf("%f, ", ref_result[i][j]);

				}
				printf("\n");
			}


			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
				{
					if (C.rows[i].vec[j] != ref_result[i][j])
					{
						printf("after matrices dont match\n");
						return;
					}
				}
			printf("success\n");
		}
		
	}

	void MatMul44Test_cpu()
	{
		for (int i = 0; i < 50; i++)
		{
			srand((unsigned int)time(NULL));


			Matrix44f A, B, C;
			ZERO_FLAT_MATRIX(C);
			Matrix44f transMat;
			float x = (float)rand() / (float)(RAND_MAX / 55.0f);
			float y = (float)rand() / (float)(RAND_MAX / 55.0f);
			float z = (float)rand() / (float)(RAND_MAX / 55.0f);

			float scale = (float)rand() / (float)(RAND_MAX / 55.0f);
			MAKE_SCALE(B, scale);

			Vector3f transVec{ x, y, z };
			MAKE_TRANSLATION(A, transVec);

			Matrix44f* A_d, * B_d, * C_d;
			//cudaMalloc((void**)&d_particles, common::ParticleData::num_particles * sizeof(common::particle_t));
			cudaMalloc((void**)&A_d, sizeof(Matrix44f));
			cudaMalloc((void**)&B_d, sizeof(Matrix44f));
			cudaMalloc((void**)&C_d, sizeof(Matrix44f));

			cudaMemcpy(A_d, (void*)&A, sizeof(Matrix44f), cudaMemcpyHostToDevice);
			cudaMemcpy(B_d, (void*)&B, sizeof(Matrix44f), cudaMemcpyHostToDevice);
			cudaMemcpy(C_d, (void*)&C, sizeof(Matrix44f), cudaMemcpyHostToDevice);

			glm::mat4 ref_mats;
			//Matrix44f grid[4];


			for (int j = 0; j < 16; j++)
			{
				float x = (float)rand() / (float)(RAND_MAX / 55.0f);
				A.mat[j] = x;
			}

			for (int j = 0; j < 4; j++)
			{
				for (int k = 0; k < 4; k++)
					ref_mats[j][k] = A.rows[j].vec[k];
			}

			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
				{
					if (A.rows[i].vec[j] != ref_mats[i][j])
					{
						printf("initial matrices dont match\n");
						return;
					}
				}
			printf("success\n");

			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{

					printf("%f, ", A.rows[i].vec[j]);

				}
				printf("\n");
			}

			printf("\n----\n");
			for (int i = 0; i < 4; i++)\
			{
				for (int j = 0; j < 4; j++)
				{
					printf("%f, ", ref_mats[i][j]);

				}
				printf("\n");
			}
			printf("\n----\n");
			printf("\n----\n");

			/*MatMul44Batch_gpu CUDA_KERNEL(1, 1) (&A, &transMat, &C, 4);
			cudaDeviceSynchronize();*/
			cudaError_t cudaerr = cudaDeviceSynchronize();
			if (cudaerr != cudaSuccess)
			{
				printf("1 kernel launch failed with error \"%s\".\n",
					cudaGetErrorString(cudaerr));
				__debugbreak;
			}
			MatMul44_gpu CUDA_KERNEL(1, 1) (A_d, B_d, C_d, 4);
			//cudaDeviceSynchronize();
			cudaerr = cudaDeviceSynchronize();
			if (cudaerr != cudaSuccess)
			{
				printf("2 kernel launch failed with error \"%s\".\n",
					cudaGetErrorString(cudaerr));
				__debugbreak;
			}



			cudaMemcpy((void*)&A, A_d, sizeof(Matrix44f), cudaMemcpyDeviceToHost);
			cudaMemcpy((void*)&B, B_d, sizeof(Matrix44f), cudaMemcpyDeviceToHost);
			cudaMemcpy((void*)&C, C_d, sizeof(Matrix44f), cudaMemcpyDeviceToHost);




			glm::mat4 ref_result = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) * glm::scale(glm::mat4(1.0f), glm::vec3(scale));

			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{

					printf("%f, ", C.rows[i].vec[j]);

				}
				printf("\n");
			}

			printf("\n----\n");
			for (int i = 0; i < 4; i++)\
			{
				for (int j = 0; j < 4; j++)
				{
					printf("%f, ", ref_result[i][j]);

				}
				printf("\n");
			}


			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
				{
					if (C.rows[i].vec[j] != ref_result[i][j])
					{
						printf("after matrices dont match\n");
						return;
					}
				}
			printf("success\n");

		}
		

	}

	






	
	
	

	void MatMulTest_cpu()
	{	
		printf("matmul cpu\n");

		
		int N = 4;

		Matrix44f A, B, C;

		ZERO_FLAT_MATRIX(C);
		//Vector3f svec{ 0.5f,0.5,0.5f };
		Vector3f tvec{ 2.0f,3.0f,4.0f };

		MAKE_TRANSLATION(A, tvec);
		//MAKE_SCALE(B, svec);
		MAKE_IDENTITY(B);

	

		dim3 threadsPerBlock(N, N);
		dim3 blocksPerGrid(1, 1);
		if (N * N > 16)
		{
			threadsPerBlock.x = 16;
			threadsPerBlock.y = 16;
			blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
			blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
		}

		//MatMul_gpu CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (&A, &B, &C, N);

		cudaDeviceSynchronize();


		glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f));

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (C.mat[j + i * 4] != model[i][j])
				{
					//std::cout << myMat.mat[j + i * 4] >> '\n';
					printf("error mult\n");
				}
			}

		}

	}

	
	

	__global__ void MakeIdentity_gpu_glm(Matrix44f* dest)
	{
		printf("makeId\n");
		glm::mat4 identity = glm::mat4(1.0f);
		memcpy(dest->mat, glm::value_ptr(identity), sizeof(float) * 16);
	}
	__global__ void MakeTranslation_gpu_glm(Matrix44f* dest, const glm::vec3& vec)
	{
		glm::mat4 source = glm::translate(glm::mat4(1.0f), vec);
		memcpy(dest->mat, glm::value_ptr(source), sizeof(float) * 16);
	}
	__global__ void MakeScale_gpu_glm(Matrix44f* dest, const glm::vec3& vec)
	{
		glm::mat4 source = glm::scale(glm::mat4(1.0f), vec);
		memcpy(dest->mat, glm::value_ptr(source), sizeof(float) * 16);
	}

	__global__ void TestIdentity_gpu()
	{
		printf("test identtityg\n");

		glm::mat4 baseCase = glm::mat4(1.0f);
		Matrix44f myMat;

		MAKE_IDENTITY(myMat);
		

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (myMat.mat[j + i * 4] != baseCase[i][j])
				{
					//std::cout << myMat.mat[j + i * 4] >> '\n';
					printf("error identity\n");
				}
			}

		}

		//printf("success identity\n");

	}
	__global__ void TestTranslation_gpu()
	{
		printf("test trangs\n");

		glm::mat4 baseCase = glm::translate(glm::mat4(1.0f), glm::vec3(7.0f, 5.0f,9.0f));
		Matrix44f myMat;

		/*myMat.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });
		myMat.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });
		myMat.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });
		myMat.rows[3] = Vector4f({ 2.0f, 3.0f, 4.0f, 1.0f });*/
		Vector3f tvec{ 7.0f,5.0f,9.0f };
		MAKE_TRANSLATION(myMat, tvec);

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (myMat.mat[j + i * 4] != baseCase[i][j])
				{
					//std::cout << myMat.mat[j + i * 4] >> '\n';
					printf("error trans\n");
				}
			}
		}
	}
	__global__ void TestScale_gpu()
	{
		printf("test scaleg\n");

		glm::mat4 baseCase = glm::scale(glm::mat4(1.0f), glm::vec3(7.0f, 5.0f, 99.0f));
		Matrix44f myMat;
		Vector3f vec = Vector3f{ 7.0f,5.0f, 99.0f };
		/*vec.x = 1.0f;
		vec.y = 2.0f;
		vec.z = 3.0f;*/
		printf("x: {%f}, y: {%f}, z: {%f}\n\n", vec.vec[0], vec.vec[1], vec.vec[2]);

		//MAKE_SCALE(myMat, vec);

	/*	myMat.rows[0] = Vector4f({ 2.0f, 0.0f, 0.0f, 0.0f });
		myMat.rows[1] = Vector4f({ 0.0f, 3.0f, 0.0f, 0.0f });
		myMat.rows[2] = Vector4f({ 0.0f, 0.0f, 4.0f, 0.0f });
		myMat.rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });*/

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (myMat.mat[j + i * 4] != baseCase[i][j])
				{
					printf("error scale\n");

				}
			}
		}
	}

	

	

	void MakeIdentity_cpu(Matrix44f* dest)
	{
		printf("makeId_cpu\n");
		cudaDeviceSynchronize();

		//MakeIdentity_gpu CUDA_KERNEL(1, 1)(dest);
		//MakeIdentity_gpu (dest);
		cudaDeviceSynchronize();

	}



	void MakeTranslation_cpu(Matrix44f* dest, const glm::vec3& vec)
	{
		printf("maketrans_cpu\n");
		//cudaDeviceSynchronize();

		//MakeTranslation_gpu CUDA_KERNEL(1, 1)(dest, vec);
		//MakeTranslation_gpu (dest, vec);

		//cudaDeviceSynchronize();
	}
	void MakeScale_cpu(Matrix44f* dest, const glm::vec3& vec)
	{
		printf("makescale_cpu\n");
		//cudaDeviceSynchronize();

		//MakeScale_gpu (dest, vec);
		//MakeScale_gpu CUDA_KERNEL(1, 1)(dest, vec);
		//cudaDeviceSynchronize();
	}

	void TestIdentity_cpu()
	{
		printf("test id cpu\n");
		cudaDeviceSynchronize();

		TestIdentity_gpu CUDA_KERNEL(1, 1)();
		cudaDeviceSynchronize();


	}
	void TestTranslation_cpu()
	{
		printf("test trans cpu\n");
		cudaDeviceSynchronize();

		TestTranslation_gpu CUDA_KERNEL(1, 1)();
		cudaDeviceSynchronize();
	}
	void TestScale_cpu()
	{
		printf("test scale cpu\n");
		cudaDeviceSynchronize();

		TestScale_gpu CUDA_KERNEL(1, 1)();
		cudaDeviceSynchronize();
	}
}