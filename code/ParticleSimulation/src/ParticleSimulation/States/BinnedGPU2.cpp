#include "sapch.h"
#include "BinnedGPU2.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "CudaSrc/Physics.cuh"



namespace ParticleSimulation {
	

	BinnedGPU2::BinnedGPU2()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}
	
	BinnedGPU2::BinnedGPU2(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
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

	void BinnedGPU2::OnEnter()
	{
		cudaError_t cudaerr;

		cudaDeviceSynchronize();

		int n = common::ParticleData::num_particles;
		float cutoff = common::ParticleData::cutoff;
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		
		//Novaura::InputHandler::GetCurrentController().BindActionInputEvent(GLFW_PRESS, GLFW_KEY_ESCAPE, &BinnedGPU2::Pause, this);
		
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
		
		grid = (common::Block*)malloc(blocks_per_side * blocks_per_side * sizeof(common::Block));
		if (grid != NULL)
		{
			for (int i = 0; i < blocks_per_side * blocks_per_side; i++) {
				grid[i].pcount = 0;
			}
		}
		else
		{
			spdlog::error("bad malloc");
			DebugBreak;
		}
		
		
		// fill the grid
		if (particles != NULL && grid!= NULL)
		{
			for (size_t i = 0; i < n; i++) {
				int block_x = (int)(particles[i].x / block_size);
				int block_y = (int)(particles[i].y / block_size);
				common::push_particle(grid[block_x * blocks_per_side + block_y], &particles[i], i);
			}
		}
		else
		{
			spdlog::error("bad fill grid");
			DebugBreak;

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


		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;
	}

	void BinnedGPU2::HandleInput()
	{
	}

	void BinnedGPU2::Update(float deltaTime)
	{		
		//spdlog::info(__FUNCTION__);
		if (m_StateInfo.RESET)
		{
			//m_StateMachine->ReplaceCurrentState(std::make_unique<ParticleSimulation::BinnedGPU2>());
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

			Physics::check_move(blks, NUM_THREADS, grid_gpu, blocks_per_side, block_size);

			cudaDeviceSynchronize();

			cudaMemcpy(particles, particles_gpu, common::ParticleData::num_particles * sizeof(common::particle_t), cudaMemcpyDeviceToHost);

			for (int i = 0; i < blocks_per_side; i++) {
				for (int j = 0; j < blocks_per_side; j++) {
					common::Block* curr = &grid[i * blocks_per_side + j];
					for (int k = 0; k < curr->pcount; k++) {
						//spdlog::info(__FUNCTION__);

						common::particle_t p = particles[curr->ids[k]];
						particles[curr->ids[k]] = *curr->particles[k];

						if (p.x != particles[curr->ids[k]].x || p.y != particles[curr->ids[k]].y)
							spdlog::info("changed!");
					}
				}
			}	
			cudaDeviceSynchronize();
			

			/*spdlog::info("p0: x {}, y {}", particles[0].x, particles[0].y);
			spdlog::info("p1: x {}, y {}", particles[1].x, particles[1].y);
			spdlog::info("p2: x {}, y {}", particles[2].x, particles[2].y);
			spdlog::info("p3: x {}, y {}", particles[3].x, particles[3].y);*/
		}

	

	}

	void BinnedGPU2::Draw(float deltaTime)
	{
		//spdlog::info(__FUNCTION__);

		Novaura::BatchRenderer::SetClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		Novaura::BatchRenderer::Clear();
		Novaura::BatchRenderer::BeginScene(m_CameraController->GetCamera());	
		m_Gui->BeginFrame();

		float width = Novaura::InputHandler::GetCurrentWindow()->Width;
		float height = Novaura::InputHandler::GetCurrentWindow()->Height;
		float aspectRatio = Novaura::InputHandler::GetCurrentWindow()->AspectRatio;		
		//float scale = common::ParticleData::density * 60.0f;
		//Novaura::BatchRenderer::StencilDraw(glm::vec3(0.5f, 0.5f, 0.0f), glm::vec3(0.25f, 0.25f, 0.0f), glm::vec4(0.5f, 0.1f, 0.8f, 1.0f), glm::vec4(0.1f, 0.8f, 0.1f, 1.0f));
		//Novaura::BatchRenderer::StencilDraw(glm::vec3(0.5f, 0.0f, 0.0f), glm::vec3(2.0f, 1.5f, 0.0f), glm::vec4(1.0f, 1.0f, 1.0f, 0.0f), glm::vec4(1.0f));
		Novaura::BatchRenderer::DrawRectangle(glm::vec3(-0.65f, 0.1f, 0.0f), glm::vec3(common::ParticleData::size, common::ParticleData::size, 1.0), glm::vec4(0.2f, 0.2f, 0.8f, 1.0f));
		for (int i = 0; i < common::ParticleData::num_particles; i++)
		{
			Novaura::BatchRenderer::DrawCircle(glm::vec3(particles[i].x -1.5f, particles[i].y-0.75, 0), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
		}

		
		/*for (auto& object : m_ObjectManager->GetCharacterList())
		{
			if(object->IsAlive())
				Novaura::BatchRenderer::DrawRotatedRectangle(object->GetRectangle(), object->GetTextureFile());
		}	*/	

		Novaura::BatchRenderer::EndScene();
		m_Gui->DrawStateButtons(m_StateInfo, particleScale);
		//m_Gui->Draw();
		//m_Gui->DrawDockSpace(m_StateInfo);
		m_Gui->EndFrame();
	}

	

	void BinnedGPU2::OnExit()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		free(particles);
		free(grid);
		cudaFree(grid_gpu);
	}

	void BinnedGPU2::Pause()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		//m_StateMachine->PushState(std::make_unique<PauseMenu>(m_Window, m_CameraController, m_StateMachine));
		
	}

	void BinnedGPU2::Resume()
	{
		//Novaura::InputHandler::SetCurrentController(m_InputController);
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}

}
