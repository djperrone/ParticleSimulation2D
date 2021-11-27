#include "sapch.h"
#include "BinnedGPU.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "CudaSrc/Physics.cuh"



namespace ParticleSimulation {
	

	BinnedGPU::BinnedGPU()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}
	
	BinnedGPU::BinnedGPU(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
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

	void BinnedGPU::OnEnter()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		
		//Novaura::InputHandler::GetCurrentController().BindActionInputEvent(GLFW_PRESS, GLFW_KEY_ESCAPE, &BinnedGPU::Pause, this);
		

		
		

		particles = (common::particle_t*)malloc(common::ParticleData::num_particles * sizeof(common::particle_t));
		common::set_size_for_gpu(common::ParticleData::num_particles);

		common::init_particles(common::ParticleData::num_particles, particles);

		m_BlocksPerSide = (int)round(sqrt(common::ParticleData::num_particles));
		m_BlockSize = common::ParticleData::size / m_BlocksPerSide;
		if (m_BlockSize < .01) {
			m_BlocksPerSide = (int)common::ParticleData::size / common::ParticleData::cutoff;
			m_BlockSize = common::ParticleData::size / m_BlocksPerSide;
		}
		
		m_Grid = (common::Block*)malloc(m_BlocksPerSide * m_BlocksPerSide * sizeof(common::Block));
		for (int i = 0; i < m_BlocksPerSide * m_BlocksPerSide; i++) {
			m_Grid[i].pcount = 0;
		}

		// fill the grid
		for (int i = 0; i < common::ParticleData::num_particles; i++) {
			int block_x = (int)(particles[i].x / m_BlockSize);
			int block_y = (int)(particles[i].y / m_BlockSize);
			common::push_particle(m_Grid[block_x * m_BlocksPerSide + block_y], &particles[i], i);
		}
		//cudaError_t cudaerr = cudaDeviceSynchronize();

		/*if (cudaerr != cudaSuccess)
		{
			printf("1 kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
			__debugbreak;
		}*/
		
		
		


		if (cudaMalloc((void**)&m_Grid_gpu, m_BlocksPerSide * m_BlocksPerSide * sizeof(common::Block)) != cudaSuccess)
		{
			printf("error cudamalloc\n");
		}
		
		if (cudaMemcpy(m_Grid_gpu, m_Grid, m_BlocksPerSide * m_BlocksPerSide * sizeof(common::Block), cudaMemcpyHostToDevice) != cudaSuccess)
		{
			printf("error cudacopy\n");

		}
		//cudaerr = cudaDeviceSynchronize();

		/*if (cudaerr != cudaSuccess)
		{
			printf("2 kernel launch failed with error \"%s\".\n",
				cudaGetErrorString(cudaerr));
			__debugbreak;
		}*/

		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;
	}

	void BinnedGPU::HandleInput()
	{
	}

	void BinnedGPU::Update(float deltaTime)
	{		
		//spdlog::info(__FUNCTION__);
		if (m_StateInfo.RESET)
		{
			//m_StateMachine->ReplaceCurrentState(std::make_unique<ParticleSimulation::BinnedGPU>());
			OnExit();
			OnEnter();
		}
		m_CameraController->Update(*Novaura::InputHandler::GetCurrentWindow(), deltaTime);
		if (!m_StateInfo.PAUSE)
		{
			//cudaError_t cudaerr = cudaDeviceSynchronize();

		/*	if (cudaerr != cudaSuccess)
			{
				printf("3 kernel launch failed with error \"%s\".\n",
					cudaGetErrorString(cudaerr));
				__debugbreak;
			}*/
			//cudaDeviceSynchronize();

			int blks = (m_BlocksPerSide * m_BlocksPerSide + NUM_THREADS - 1) / NUM_THREADS;
			//Physics::compute_forces_gpu << < blks, NUM_THREADS >> > (d_particles, n);
			Physics::compute_forces(blks, NUM_THREADS, m_Grid_gpu, m_BlocksPerSide);
			//cudaError_t cudaerr = cudaDeviceSynchronize();
			//cudaerr = cudaDeviceSynchronize();
			/*if (cudaerr != cudaSuccess)
			{
				printf("4 kernel launch failed with error \"%s\".\n",
					cudaGetErrorString(cudaerr));
				__debugbreak;
			}*/
				
			//Physics::move(blks, NUM_THREADS, m_Grid_gpu, m_BlocksPerSide, common::ParticleData::size);
			//Physics::check_move(blks, NUM_THREADS, m_Grid_gpu, m_BlocksPerSide, m_BlockSize);

			//cudaDeviceSynchronize();

			//cudaMemcpy(m_Grid, m_Grid_gpu, m_BlocksPerSide * m_BlocksPerSide * sizeof(common::Block), cudaMemcpyDeviceToHost);

			//for (int i = 0; i < m_BlocksPerSide; i++) {
			//	for (int j = 0; j < m_BlocksPerSide; j++) {
			//		common::Block& curr = m_Grid[i * m_BlocksPerSide + j];
			//		for (int k = 0; k < curr.pcount; k++) {
			//			//spdlog::info(__FUNCTION__);

			//			common::particle_t p = particles[curr.ids[k]];
			//			particles[curr.ids[k]] = *curr.particles[k];

			//			if (p.x != particles[curr.ids[k]].x || p.y != particles[curr.ids[k]].y)
			//				spdlog::info("changed!");
			//		}
			//	}
			//}	
			//cudaDeviceSynchronize();
			

			/*spdlog::info("p0: x {}, y {}", particles[0].x, particles[0].y);
			spdlog::info("p1: x {}, y {}", particles[1].x, particles[1].y);
			spdlog::info("p2: x {}, y {}", particles[2].x, particles[2].y);
			spdlog::info("p3: x {}, y {}", particles[3].x, particles[3].y);*/
		}

	

	}

	void BinnedGPU::Draw(float deltaTime)
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

	

	void BinnedGPU::OnExit()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		free(particles);
		free(m_Grid);
		cudaFree(m_Grid_gpu);
	}

	void BinnedGPU::Pause()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		//m_StateMachine->PushState(std::make_unique<PauseMenu>(m_Window, m_CameraController, m_StateMachine));
		
	}

	void BinnedGPU::Resume()
	{
		//Novaura::InputHandler::SetCurrentController(m_InputController);
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}

}
