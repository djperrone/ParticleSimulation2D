//#include "CudaMath.h"
#include "CUDA_KERNEL.h"
#include "Math/Matrix.h"

#define ZERO_FLAT_MATRIX(mat)  memset(mat, 0, 16 * sizeof(float));

namespace CudaMath {

	typedef union 
	{
		float vec[3];

		struct
		{
			float x, y, z;
		};
	}Vector3f;

	typedef union  
	{
		float vec[4];
		struct
		{
			float x, y, z, w;
		};
	}Vector4f;

	typedef union   
	{
		float mat[16];
		Vector4f rows[4];	
	}FlatMatrix;


	//__global__ void MatMul_gpu(Math::FlatMatrix* A, Math::FlatMatrix* B, Math::FlatMatrix* C, int N);
	void MatMul_cpu();
	__global__ void MatMulTest_gpu();
	__global__ void MatMulTest_cpu();

	//__global__ void MatMulVec();


	


	__device__ void MakeIdentity_gpu(FlatMatrix* dest);
	__device__  void MakeScale_gpu(FlatMatrix* dest, const Vector3f& vec);
	__device__ void MakeTranslation_gpu(FlatMatrix* dest, const Vector3f& vec);

	__global__ void MakeIdentity_gpu_glm(FlatMatrix* dest);
	__global__ void MakeTranslation_gpu_glm(FlatMatrix* dest, const glm::vec3& vec);
	__global__ void MakeScale_gpu_glm(FlatMatrix* dest, const glm::vec3& vec);

	__global__ void TestIdentity_gpu();
	__global__ void TestTranslation_gpu();
	__global__ void TestScale_gpu();

	void MakeIdentity_cpu(FlatMatrix* dest);

	void MakeTranslation_cpu(FlatMatrix* dest, const Vector3f& vec);
	void MakeScale_cpu(FlatMatrix* dest, const Vector3f& vec);

	void TestIdentity_cpu();
	void TestTranslation_cpu();
	void TestScale_cpu();



	

}
