#include "CudaMath.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace CudaMath {

	
	__global__ void MatMul_gpu(FlatMatrix* A, FlatMatrix* B, FlatMatrix* C, int N)
	{
		printf("matmul gpu\n");
		int ROW = blockIdx.y * blockDim.y + threadIdx.y;
		int COL = blockIdx.x * blockDim.x + threadIdx.x;
		float tmpSum = 0;

		if (ROW < N && COL < N) {
			// each thread computes one element of the block sub-matrix
			for (int i = 0; i < N; i++) {
				tmpSum += A->mat[ROW * N + i] * B->mat[i * N + COL];
			}
		}
		C->mat[ROW * N + COL] = tmpSum;

	}

	__global__ void MatMulTest_gpu()
	{

		/*int N = 4;

		FlatMatrix A, B, C;

		ZERO_FLAT_MATRIX(C);

		MakeTranslation_gpu(&A, Vector3f({ 2.0f,3.0f,4.0f }));
		MakeScale_gpu(&A, Vector3f({ 0.5f,0.5,0.5f }));

		dim3 threadsPerBlock(N, N);
		dim3 blocksPerGrid(1, 1);
		if (N * N > 16)
		{
			threadsPerBlock.x = 16;
			threadsPerBlock.y = 16;
			blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
			blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
		}

		MatMul_gpu CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (&A, &B, &C, N);*/
		//cudaDeviceSynchronize();


		//glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 0.5f, 0.5f));

		//for (int i = 0; i < 4; i++)
		//{
		//	for (int j = 0; j < 4; j++)
		//	{
		//		if (C.mat[j + i * 4] != model[i][j])
		//		{
		//			//std::cout << myMat.mat[j + i * 4] >> '\n';
		//			printf("error identity\n");
		//		}
		//	}

		//}

	}
	
	

	void MatMulTest_cpu()
	{	
		printf("matmul cpu\n");

		/*cudaDeviceSynchronize();

		MatMulTest_gpu CUDA_KERNEL(1, 1)();
		cudaDeviceSynchronize();*/

		int N = 4;

		FlatMatrix A, B, C;

		ZERO_FLAT_MATRIX(C);
		//Vector3f svec{ 0.5f,0.5,0.5f };
		Vector3f tvec{ 2.0f,3.0f,4.0f };

		MAKE_TRANSLATION(A, tvec);
		//MAKE_SCALE(B, svec);
		MAKE_IDENTITY(B);

		//MakeTranslation_gpu(&A, );
		//MakeScale_gpu(&A, Vector3f({ 0.5f,0.5,0.5f }));

		/*A.rows[0] = Vector4f({ svec.x, 0.0f, 0.0f, 0.0f });
		A.rows[1] = Vector4f({ 0.0f, svec.y, 0.0f, 0.0f });
		A.rows[2] = Vector4f({ 0.0f, 0.0f, svec.z, 0.0f });
		A.rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });*/

		dim3 threadsPerBlock(N, N);
		dim3 blocksPerGrid(1, 1);
		if (N * N > 16)
		{
			threadsPerBlock.x = 16;
			threadsPerBlock.y = 16;
			blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
			blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
		}

		MatMul_gpu CUDA_KERNEL(blocksPerGrid, threadsPerBlock) (&A, &B, &C, N);

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

	
	//__device__ void CudaMath::MakeIdentity_gpu(FlatMatrix* dest)
	//{
	//	dest->rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });
	//	dest->rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });
	//	dest->rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });
	//	dest->rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });
	//}
	//__device__ void MakeScale_gpu(FlatMatrix* dest, const Vector3f& vec)
	//{
	//	dest->rows[0] = Vector4f({ vec.x, 0.0f, 0.0f, 0.0f });
	//	dest->rows[1] = Vector4f({ 0.0f, vec.y, 0.0f, 0.0f });
	//	dest->rows[2] = Vector4f({ 0.0f, 0.0f, vec.z, 0.0f });
	//	dest->rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

	//	printf("[] x: {%f}, y: {%f}, z: {%f}", vec.vec[0], vec.vec[1], vec.vec[2]);
	//	printf("x: {%f}, y: {%f}, z: {%f}", vec.x, vec.y, vec.z);
	//}

	//__device__ void MakeTranslation_gpu(FlatMatrix* dest, const Vector3f& vec)
	//{
	//	dest->rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });
	//	dest->rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });
	//	dest->rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });
	//	dest->rows[3] = Vector4f({ vec.x, vec.y, vec.z, 1.0f });
	//}

	__global__ void MakeIdentity_gpu_glm(FlatMatrix* dest)
	{
		printf("makeId\n");
		glm::mat4 identity = glm::mat4(1.0f);
		memcpy(dest->mat, glm::value_ptr(identity), sizeof(float) * 16);
	}
	__global__ void MakeTranslation_gpu_glm(FlatMatrix* dest, const glm::vec3& vec)
	{
		glm::mat4 source = glm::translate(glm::mat4(1.0f), vec);
		memcpy(dest->mat, glm::value_ptr(source), sizeof(float) * 16);
	}
	__global__ void MakeScale_gpu_glm(FlatMatrix* dest, const glm::vec3& vec)
	{
		glm::mat4 source = glm::scale(glm::mat4(1.0f), vec);
		memcpy(dest->mat, glm::value_ptr(source), sizeof(float) * 16);
	}

	__global__ void TestIdentity_gpu()
	{
		printf("test identtityg\n");

		glm::mat4 baseCase = glm::mat4(1.0f);
		FlatMatrix myMat;

		MAKE_IDENTITY(myMat);
		/*myMat.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });
		myMat.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });
		myMat.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });
		myMat.rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });*/

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
		FlatMatrix myMat;

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
		FlatMatrix myMat;
		Vector3f vec = Vector3f{ 7.0f,5.0f, 99.0f };
		/*vec.x = 1.0f;
		vec.y = 2.0f;
		vec.z = 3.0f;*/
		printf("x: {%f}, y: {%f}, z: {%f}\n\n", vec.vec[0], vec.vec[1], vec.vec[2]);

		MAKE_SCALE(myMat, vec);

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

	

	void MakeIdentity_cpu(FlatMatrix* dest)
	{
		printf("makeId_cpu\n");
		cudaDeviceSynchronize();

		//MakeIdentity_gpu CUDA_KERNEL(1, 1)(dest);
		//MakeIdentity_gpu (dest);
		cudaDeviceSynchronize();

	}

	void MakeTranslation_cpu(FlatMatrix* dest, const Vector3f& vec)
	{
	}

	void MakeScale_cpu(FlatMatrix* dest, const Vector3f& vec)
	{
	}

	void MakeTranslation_cpu(FlatMatrix* dest, const glm::vec3& vec)
	{
		printf("maketrans_cpu\n");
		//cudaDeviceSynchronize();

		//MakeTranslation_gpu CUDA_KERNEL(1, 1)(dest, vec);
		//MakeTranslation_gpu (dest, vec);

		//cudaDeviceSynchronize();
	}
	void MakeScale_cpu(FlatMatrix* dest, const glm::vec3& vec)
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