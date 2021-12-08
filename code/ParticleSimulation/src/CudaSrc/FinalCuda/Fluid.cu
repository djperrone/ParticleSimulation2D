#include "Fluid.cuh"
#include "Novaura/CudaGLInterop/helper_cuda.h"
#include "utilities.cuh"



namespace StableFluidsCuda {

    void FluidSquareCreate(FluidSquare* sq, int size, float diffusion, float viscosity, float dt) {
      
        //FluidData* data = nullptr;
       // checkCudaErrors(cudaMalloc((void**)data, sizeof(FluidData)));
        FluidData localData = { size, diffusion, viscosity, dt };
        //checkCudaErrors(cudaMemcpy(data, &local, sizeof(FluidData), cudaMemcpyHostToDevice));
        printf("%i\n", __LINE__);

        sq->data = { size, diffusion, viscosity, dt };

       
       
        printf("%i\n",__LINE__);
        int N = size;       
        printf("%i\n", __LINE__);

        float* temp;
        checkCudaErrors(cudaMalloc((void**)&sq->density0, sizeof(float) * N * N));
        printf("%i\n", __LINE__);

        checkCudaErrors(cudaMalloc((void**)&sq->density, sizeof(float) * N * N));
        printf("%i\n", __LINE__);

        checkCudaErrors(cudaMalloc((void**)&sq->Vx, sizeof(float) * N * N));
        checkCudaErrors(cudaMalloc((void**)&sq->Vy, sizeof(float) * N * N));

        checkCudaErrors(cudaMalloc((void**)&sq->Vx0, sizeof(float) * N * N));
        checkCudaErrors(cudaMalloc((void**)&sq->Vy0, sizeof(float) * N * N));
        printf("%i\n", __LINE__);

       // return sq;
    }

    void FluidSquareCreate_cpu(FluidSquare* sq, int size, float diffusion, float viscosity, float dt)
    {
        int N = size;

        sq->data = { size, diffusion, viscosity, dt };
        sq->density0 = (float*)calloc(N * N, sizeof(float));
        sq->density = (float*)calloc(N * N, sizeof(float));

        sq->Vx = (float*)calloc(N * N, sizeof(float));
        sq->Vy = (float*)calloc(N * N, sizeof(float));

        sq->Vx0 = (float*)calloc(N * N, sizeof(float));
        sq->Vy0 = (float*)calloc(N * N, sizeof(float));

    }

    FluidSquare* FluidSquareCreate_cpu(int size, float diffusion, float viscosity, float dt)
    {
        FluidSquare* sq = (FluidSquare*)malloc(sizeof(FluidSquare));
        int N = size;

        sq->data = { size, diffusion, viscosity, dt };

        sq->density0 = (float*)calloc(N * N, sizeof(float));
        sq->density = (float*)calloc(N * N, sizeof(float));

        sq->Vx = (float*)calloc(N * N, sizeof(float));
        sq->Vy = (float*)calloc(N * N, sizeof(float));

        sq->Vx0 = (float*)calloc(N * N, sizeof(float));
        sq->Vy0 = (float*)calloc(N * N, sizeof(float));

        return sq;
    }

    void CopyToCPU(FluidSquare* sq_cpu, FluidSquare* sq_gpu, int N)
    {
        cudaMemcpy(sq_cpu->density0, sq_gpu->density0, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sq_cpu->density, sq_gpu->density, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        cudaMemcpy(sq_cpu->Vx, sq_gpu->Vx, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sq_cpu->Vy, sq_gpu->Vy, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        cudaMemcpy(sq_cpu->Vx0, sq_gpu->Vx0, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(sq_cpu->Vy0, sq_gpu->Vy0, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        cudaMemcpy(&sq_cpu->data, &sq_gpu->data, sizeof(FluidData), cudaMemcpyDeviceToHost);
    }

    void FluidSquareFree(FluidSquare* sq) {      

        checkCudaErrors(cudaFree(sq->density0));
        checkCudaErrors(cudaFree(sq->density));

        checkCudaErrors(cudaFree(sq->Vx));
        checkCudaErrors(cudaFree(sq->Vy));

        checkCudaErrors(cudaFree(sq->Vx0));
        checkCudaErrors(cudaFree(sq->Vy0));

        //checkCudaErrors(cudaFree(sq));
    }

    void FluidSquareFree_cpu(FluidSquare* sq)
    {
        free(sq->density0);
        free(sq->density);

        free(sq->Vx);
        free(sq->Vy);

        free(sq->Vx0);
        free(sq->Vy0);

        free(sq);
    }

    __global__ void FluidSquareAddDensity_gpu(FluidSquare* sq, int x, int y, float amount)
    {
        int N = sq->data.size;
        int index = IX(x, y);

        if (index < 0 || index >= N * N) return;
       
        sq->density[IX(x, y)] += amount;
    }

    __global__ void FluidSquareAddVelocity_gpu(FluidSquare* sq, int x, int y, float amountX, float amountY)
    {
        int N = sq->data.size;
        int index = IX(x, y);
        
        if (index < 0 || index >= N * N) return;

        sq->Vx[index] += amountX;
        sq->Vy[index] += amountY;
    }

    void FluidSquareAddDensity(FluidSquare* sq, int x, int y, float amount) {

        FluidSquareAddDensity_gpu CUDA_KERNEL(1, 1)(sq, x, y, amount);
    }

    void FluidSquareAddVelocity(FluidSquare* sq, int x, int y, float amountX, float amountY) {       

        FluidSquareAddVelocity_gpu CUDA_KERNEL(1, 1)(sq, x, y, amountX, amountY);
    }

    void FluidSquareStep(FluidSquare* sq) {

        int N = sq->data.size;
        float visc = sq->data.visc;
        float diff = sq->data.diff;
        float dt = sq->data.dt;
        float* Vx = sq->Vx;
        float* Vy = sq->Vy;
        float* Vx0 = sq->Vx0;
        float* Vy0 = sq->Vy0;
        float* density0 = sq->density0;
        float* density = sq->density;

        int blks = (N * N + NUM_THREADS - 1) / NUM_THREADS;


        diffuse_gpu CUDA_KERNEL (blks, NUM_THREADS)  (1, Vx0, Vx, visc, dt, 4, N);
        diffuse_gpu CUDA_KERNEL (blks, NUM_THREADS) (2, Vy0, Vy, visc, dt, 4, N);

        project_gpu CUDA_KERNEL(blks, NUM_THREADS) (Vx0, Vy0, Vx, Vy, 4, N);

        advect_gpu CUDA_KERNEL (blks, NUM_THREADS) (1, Vx, Vx0, Vx0, Vy0, dt, N);
        advect_gpu CUDA_KERNEL (blks, NUM_THREADS) (2, Vy, Vy0, Vx0, Vy0, dt, N);

        project_gpu CUDA_KERNEL(blks, NUM_THREADS) (Vx, Vy, Vx0, Vy0, 4, N);

        diffuse_gpu CUDA_KERNEL (blks, NUM_THREADS) (0, density0, density, diff, dt, 4, N);
        advect_gpu CUDA_KERNEL (blks, NUM_THREADS) (0, density, density0, Vx, Vy, dt, N);
    }
}