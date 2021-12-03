#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cudagl.h>

namespace CudaGLInterop {

	void InitDevices();

	void SetDefaultCudaDevice();
	void RegisterCudaGLBuffer(cudaGraphicsResource* positionsVBO_CUDA, unsigned int positionsVBO);

	void MapCudaGLMatrixBuffer(cudaGraphicsResource* positionsVBO_CUDA, size_t* num_bytes, glm::mat4* matrices);

	void UnMapCudaGLMatrixBuffer(cudaGraphicsResource* positionsVBO_CUDA);

}

