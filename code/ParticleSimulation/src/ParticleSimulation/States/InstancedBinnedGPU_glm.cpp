#include "sapch.h"
#include "InstancedBinnedGPU_glm.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "CudaSrc/Physics.cuh"
#include "CudaSrc/CudaMath.cuh"

//#include <cuda_gl_interop.h>
//#include <cudagl.h>

// need to re-register cuda device only on resize
// dont need to change vertex array on resize
// break up ininit interop to a few functions to avoid unecesary allocations

namespace ParticleSimulation {
	

	InstancedBinnedGPU_glm::InstancedBinnedGPU_glm()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}
	
	InstancedBinnedGPU_glm::InstancedBinnedGPU_glm(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
		: m_StateInfo()
	{
		m_Window = window;
		m_CameraController = cameraController;
		m_StateMachine = stateMachine;
		m_Gui = std::make_unique<Pgui::Gui>(Novaura::InputHandler::GetCurrentWindow());
		m_InputController = Novaura::InputHandler::CreateNewInputController();
		Novaura::InputHandler::SetCurrentController(m_InputController);
		OnEnter();
	}

	void InstancedBinnedGPU_glm::OnEnter()
	{
		cudaError_t cudaerr;

		cudaDeviceSynchronize();

		int n = common::ParticleData::num_particles;
		float cutoff = common::ParticleData::cutoff;		
		
		particles = (common::particle_t*)malloc(n * sizeof(common::particle_t));
		if (particles == NULL)
		{
			spdlog::error("bad malloc");
			DebugBreak;
		}
		common::set_size_for_gpu(n);
		double size = common::ParticleData::size;
		common::init_particles(n, particles);

		blocks_per_side = (int)round(sqrt(n));
		block_size = size / blocks_per_side;
		if (block_size < .01) {
			blocks_per_side = (int)size / cutoff;
			block_size = size / blocks_per_side;
		}			

		cudaDeviceSynchronize();
		if (blocks_per_side != 0)
		{
			/*cudaMalloc((void**)&grid_gpu, blocks_per_side * blocks_per_side * sizeof(common::Block));*/
			if (cudaMalloc((void**)&grid_gpu, blocks_per_side * blocks_per_side * sizeof(common::Block)) != cudaSuccess)
			{
				spdlog::error("error cudamalloc\n");
			}
		}
		else
		{
			spdlog::error("bad cuda malloc size 0");
			DebugBreak;
		}		

		
		cudaMalloc((void**)&particles_gpu, n * sizeof(common::particle_t));
		cudaMemcpy(particles_gpu, particles, n * sizeof(common::particle_t), cudaMemcpyHostToDevice);

		cudaerr = cudaDeviceSynchronize();

		if (cudaerr != cudaSuccess)
		{
			printf("5kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
			__debugbreak;
		}

		//cudaMemcpy(grid_gpu, grid, blocks_per_side * blocks_per_side * sizeof(common::Block), cudaMemcpyHostToDevice);

		cudaerr = cudaDeviceSynchronize();

		if (cudaerr != cudaSuccess)
		{
			printf("6 kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
			__debugbreak;
		}

		spdlog::info("init from binnedgpu");
		
		Physics::InitGrid(grid_gpu, particles_gpu, blocks_per_side, block_size, n);
		//Physics::Test();
		cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("7 kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
			__debugbreak;
			exit(-1);
		}
		Novaura::Renderer::InitInteropInstancedCircles_glm(common::ParticleData::num_particles, particleScale, glm::vec4(0.8f, 0.2f, 0.2f, 1.0f));
		Novaura::Renderer::UpdateMatricesInterop_glm(particles_gpu, common::ParticleData::num_particles);


		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;

		//CudaMath::MatMul44Test_cpu();
	/*	CudaMath::MatMul44BatchTest_cpu();
		exit(2);*/
	}

	void InstancedBinnedGPU_glm::HandleInput()
	{
	}

	void InstancedBinnedGPU_glm::Update(float deltaTime)
	{		
		//spdlog::info(__FUNCTION__);
		if (m_StateInfo.RESET)
		{
			//m_StateMachine->ReplaceCurrentState(std::make_unique<ParticleSimulation::InstancedBinnedGPU_glm>());
			OnExit();
			OnEnter();
		}
		m_CameraController->Update(*Novaura::InputHandler::GetCurrentWindow(), deltaTime);
		if (!m_StateInfo.PAUSE)
		{


			int blks = (blocks_per_side * blocks_per_side + NUM_THREADS - 1) / NUM_THREADS;
			//Physics::compute_forces_gpu << < blks, NUM_THREADS >> > (d_particles, n);
			Physics::compute_forces(blks, NUM_THREADS, grid_gpu, blocks_per_side);
			cudaDeviceSynchronize();


			Physics::move(blks, NUM_THREADS, grid_gpu, blocks_per_side, common::ParticleData::size);
			cudaDeviceSynchronize();

			//Physics::check_move_wrapper(blks, NUM_THREADS, grid_gpu, blocks_per_side, block_size);
			//check_move_serial(common::Block * grid, common::particle_t * particles_gpu, size_t num_particles, size_t blocks_per_side, double block_size)
			Physics::check_move_serial(grid_gpu, particles_gpu, common::ParticleData::num_particles, blocks_per_side, block_size);


			/*cudaMemcpy(particles, particles_gpu, common::ParticleData::num_particles * sizeof(common::particle_t), cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();	*/
		}
	}

	void InstancedBinnedGPU_glm::Draw(float deltaTime)
	{
		//spdlog::info(__FUNCTION__);

		Novaura::Renderer::SetClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		Novaura::Renderer::Clear();
		Novaura::Renderer::BeginSceneInstanced(m_CameraController->GetCamera());
		m_Gui->BeginFrame();

		float width = Novaura::InputHandler::GetCurrentWindow()->Width;
		float height = Novaura::InputHandler::GetCurrentWindow()->Height;
		float aspectRatio = Novaura::InputHandler::GetCurrentWindow()->AspectRatio;		
	
		//Novaura::Renderer::DrawInstancedCircle(glm::vec3(-0.65f, 0.1f, 0.0f), glm::vec3(common::ParticleData::size, common::ParticleData::size, 1.0), glm::vec4(0.2f, 0.2f, 0.8f, 1.0f));
		/*for (int i = 0; i < common::ParticleData::num_particles; i++)
		{
			Novaura::Renderer::DrawInstancedCircle(glm::vec3(particles[i].x -1.5f, particles[i].y-0.75, 0), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
		}		*/

		Novaura::Renderer::UpdateMatricesInterop_glm(particles_gpu, common::ParticleData::num_particles);
	
		Novaura::Renderer::EndInteropInstancedCircles();

		m_Gui->DrawStateButtons(m_StateInfo, particleScale);
		
		m_Gui->EndFrame();
	}

	

	void InstancedBinnedGPU_glm::OnExit()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		free(particles);
		cudaFree(particles_gpu);
		cudaFree(grid_gpu);
	}

	void InstancedBinnedGPU_glm::Pause()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		//m_StateMachine->PushState(std::make_unique<PauseMenu>(m_Window, m_CameraController, m_StateMachine));
		
	}

	void InstancedBinnedGPU_glm::Resume()
	{
		//Novaura::InputHandler::SetCurrentController(m_InputController);
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}

	

}
