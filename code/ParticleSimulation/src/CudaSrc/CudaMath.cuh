#pragma once
//#include "CudaMath.h"
#include "CUDA_KERNEL.h"
#include "Math/Matrix.h"
#include "common/particle_t.h"

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
	}Matrix44f;

#define MAKE_IDENTITY(dest)dest.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });\
dest.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });\
dest.rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

#define MAKE_SCALE(dest, scale) \
dest.rows[0] = CudaMath::Vector4f({ scale, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = CudaMath::Vector4f({ 0.0f, scale, 0.0f, 0.0f });\
dest.rows[2] = CudaMath::Vector4f({ 0.0f, 0.0f, scale, 0.0f });\
dest.rows[3] = CudaMath::Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

#define MAKE_TRANSLATION(dest, vec) dest.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });\
dest.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });\
dest.rows[3] = Vector4f({ vec.x, vec.y, vec.z, 1.0f });

#define MAKE_TRANSLATION_xyz(dest, x,y,z) dest.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });\
dest.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });\
dest.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });\
dest.rows[3] = Vector4f({ x, y, z, 1.0f });

	__global__ void MatMul44Batch_gpu(Matrix44f* grid, Matrix44f* B, Matrix44f* C, int N);
	void MatMul44Batch_cpu(Matrix44f* grid, Matrix44f* B, Matrix44f* C, int numParticles);
	
	__global__ void MatMul44_gpu(Matrix44f* A, Matrix44f* B, Matrix44f* C, int numParticles);
	__global__ void MatMul44_gpu(Matrix44f* A, Matrix44f* B, Matrix44f* C, int* numParticles);
	void MatMul44_cpu(Matrix44f* A, Matrix44f* B, Matrix44f* C, int N);
	


	
	void __global__ MakeTranslationMatrices_gpu(Matrix44f* matrices, common::particle_t* particles, size_t numParticles);
	void MakeTranslationMatrices_cpu(Matrix44f* matrices, common::particle_t* particles, size_t numParticles);
	void UpdateMatrices_cpu(Matrix44f* matrices, common::particle_t* particles, size_t numParticles);



	__global__ void TestIdentity_gpu();
	__global__ void TestTranslation_gpu();
	__global__ void TestScale_gpu();


	void MatMul44Test_cpu();
	void MatMul44Test_cpu2();

	void MakeIdentity_cpu(Matrix44f* dest);


	void TestIdentity_cpu();
	void TestTranslation_cpu();
	void TestScale_cpu();


	void MatMul44BatchTest_cpu();


	

}
