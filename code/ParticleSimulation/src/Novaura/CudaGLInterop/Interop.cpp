#include <sapch.h>
#include "Interop.h"

#include "Novaura/Novaura.h"

namespace CudaGLInterop {


	void InitDevices()
	{
		cudaError_t cuda_err;
		unsigned int gl_device_count;
		int gl_device_id;
		int cuda_device_id;
		cuda_err = cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll);
		//SetDevice(cuda_device_id));
		cuda_err = cudaSetDevice(cuda_device_id);

		struct cudaDeviceProp props;
		cuda_err = cudaGetDeviceProperties(&props, gl_device_id);
		printf("GL   : %-24s (%2d)\n", props.name, props.multiProcessorCount);

		cuda_err = cudaGetDeviceProperties(&props, cuda_device_id);
		printf("CUDA : %-24s (%2d)\n", props.name, props.multiProcessorCount);

		cudaStream_t stream;
		cudaEvent_t  event;
		struct cudaGraphicsResource* cuda_vbo_resource;

		//cuda_err = cudaStreamCreateWithFlags(&stream, cudaStreamDefault);   // optionally ignore default stream behavior
		//cuda_err = cudaEventCreateWithFlags(&event, cudaEventBlockingSync); // | cudaEventDisableTiming);
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		//glBufferData(GL_ARRAY_BUFFER, sizeof(cData) * DS, particles,
		//	GL_DYNAMIC_DRAW_ARB);

		cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsNone);

		//cudaGLUnregisterBufferObject(vbo);

		//glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

	}

	void SetDefaultCudaDevice()
	{
		//cudaGLSetGLDevice(0);

	}

	void RegisterCudaGLBuffer(cudaGraphicsResource* positionsVBO_CUDA, unsigned int positionsVBO)
	{
		cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
	}

	void UnRegisterCudaGLBuffer(cudaGraphicsResource* positionsVBO_CUDA, unsigned int positionsVBO)
	{
		cudaGraphicsUnregisterResource(positionsVBO_CUDA);
	}

	void MapCudaGLMatrixBuffer(cudaGraphicsResource* positionsVBO_CUDA, size_t* num_bytes, glm::mat4* matrices)
	{
		cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
		//size_t num_bytes;
		cudaGraphicsResourceGetMappedPointer((void**)&matrices, num_bytes, positionsVBO_CUDA);
	}

	void UnMapCudaGLMatrixBuffer(cudaGraphicsResource* positionsVBO_CUDA)
	{
		cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
	}

}