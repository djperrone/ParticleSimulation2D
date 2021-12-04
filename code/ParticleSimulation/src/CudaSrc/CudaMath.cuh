//#include "CudaMath.h"
#include "CUDA_KERNEL.h"
#include "Math/Matrix.h"

#define ZERO_FLAT_MATRIX(Mat)  memset(Mat.mat, 0, 16 * sizeof(float));




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

#define MAKE_IDENTITY(dest)dest.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });\
dest.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });\
dest.rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

#define MAKE_SCALE(dest, scale) \
dest.rows[0] = Vector4f({ scale, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = Vector4f({ 0.0f, scale, 0.0f, 0.0f });\
dest.rows[2] = Vector4f({ 0.0f, 0.0f, scale, 0.0f });\
dest.rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

#define MAKE_TRANSLATION(dest, vec) dest.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });\
dest.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });\
dest.rows[3] = Vector4f({ vec.x, vec.y, vec.z, 1.0f });


	__global__ void MatMul44Batch_gpu(FlatMatrix* grid, FlatMatrix* B, FlatMatrix* C, int N);
	//__global__ void MatMul_gpu();
	__global__ void MatMul44_gpu(FlatMatrix* A, FlatMatrix* B, FlatMatrix* C, int N);
	void MatMul44_cpu(FlatMatrix* A, FlatMatrix* B, FlatMatrix* C, int N);
	//__global__ void MatMulTest_gpu();

	//__global__ void MatMulVec();


	




	__global__ void TestIdentity_gpu();
	__global__ void TestTranslation_gpu();
	__global__ void TestScale_gpu();


	void MatMul44Test_cpu();
	void MatMul44Test_cpu2();

	void MakeIdentity_cpu(FlatMatrix* dest);

	void MakeTranslation_cpu(FlatMatrix* dest, const Vector3f& vec);
	void MakeScale_cpu(FlatMatrix* dest, const Vector3f& vec);

	void TestIdentity_cpu();
	void TestTranslation_cpu();
	void TestScale_cpu();


	void MatMul44BatchTest_cpu();
	__global__ void MatMul44Test_gpu();

	void MulVecMat44(FlatMatrix* mat, Vector4f& vec, Vector4f* result);
	void DotProduct(Vector4f& A, Vector4f& B, Vector4f* result);

}
