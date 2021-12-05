#include "sapch.h"
#include "InstancedBinnedCPU.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "CudaSrc/Physics.cuh"
#include <glm/gtc/type_ptr.hpp>

#include "Math/Matrix.h"
#include "CudaSrc/CudaMath.cuh"
namespace ParticleSimulation {
	

	InstancedBinnedCPU::InstancedBinnedCPU()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}
	
	InstancedBinnedCPU::InstancedBinnedCPU(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
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

	InstancedBinnedCPU::~InstancedBinnedCPU()
	{
		OnExit();
	}

	void InstancedBinnedCPU::OnEnter()
	{
		Novaura::Renderer::InitInstancedCircles_glm(common::ParticleData::num_particles, particleScale, particleColor);

		particles = (common::particle_t*)malloc(common::ParticleData::num_particles * sizeof(common::particle_t));
		common::set_size(common::ParticleData::num_particles);

		common::init_particles(common::ParticleData::num_particles, particles);

		m_BlocksPerSide = (int)round(sqrt(common::ParticleData::num_particles));
		double block_size = common::ParticleData::size / m_BlocksPerSide;
		if (block_size < .01) {
			m_BlocksPerSide = (int)common::ParticleData::size / common::ParticleData::cutoff;
			block_size = common::ParticleData::size / m_BlocksPerSide;
		}		

		grid = pvec::InitGrid3d(m_BlocksPerSide);
		// fill the grid
		for (int i = 0; i < common::ParticleData::num_particles; i++) {
			int block_x = (int)particles[i].x / block_size;
			int block_y = (int)particles[i].y / block_size;
			
			pvec::PushParticle(grid[block_x][block_y], (&particles[i]));
		}

		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;

		//Math::TestIdentity();
		//Math::TestTranslation();
		//Math::TestScale();

		//CudaMath::TestIdentity_cpu();
		//CudaMath::TestTranslation_cpu();
		//CudaMath::TestScale_cpu();

		//CudaMath::MatMulTest_cpu();

	}

	void InstancedBinnedCPU::HandleInput()
	{
	}

	void InstancedBinnedCPU::Update(float deltaTime)
	{		
		if (m_StateInfo.RESET)
		{
			//m_StateMachine->ReplaceCurrentState(std::make_unique<ParticleSimulation::InstancedBinnedCPU>());
			OnExit();
			OnEnter();
		}
		m_CameraController->Update(*Novaura::InputHandler::GetCurrentWindow(), deltaTime);
		if (!m_StateInfo.PAUSE)
		{
			int blks = (common::ParticleData::num_particles + NUM_THREADS - 1) / NUM_THREADS;
			//Physics::compute_forces_gpu << < blks, NUM_THREADS >> > (d_particles, n);

			for (int i = 0; i < m_BlocksPerSide; i++) {
				for (int j = 0; j < m_BlocksPerSide; j++) {

					// set acceleration of particles in block to 0
					for (int k = 0; k < grid[i][j].pcount; k++) {
						grid[i][j].data[k]->ax = grid[i][j].data[k]->ay = 0;
					}

					// check each of 8 neighbors exists, calling apply_across_blocks if so
					if (j != m_BlocksPerSide - 1) { //right
						Physics::apply_across_blocks(grid[i][j], grid[i][j + 1]);
					}
					if (j != m_BlocksPerSide - 1 && i != m_BlocksPerSide - 1) { //down+right
						Physics::apply_across_blocks(grid[i][j], grid[i + 1][j + 1]);
					}
					if (j != m_BlocksPerSide - 1 && i != 0) { //up+right
						Physics::apply_across_blocks(grid[i][j], grid[i - 1][j + 1]);
					}
					if (i != 0) { //up
						Physics::apply_across_blocks(grid[i][j], grid[i - 1][j]);
					}
					if (i != m_BlocksPerSide - 1) { //down
						Physics::apply_across_blocks(grid[i][j], grid[i + 1][j]);
					}
					if (j != 0) { //left
						Physics::apply_across_blocks(grid[i][j], grid[i][j - 1]);
					}
					if (j != 0 && i != 0) { //up+left
						Physics::apply_across_blocks(grid[i][j], grid[i - 1][j - 1]);
					}
					if (j != 0 && i != m_BlocksPerSide - 1) { //down+left
						Physics::apply_across_blocks(grid[i][j], grid[i + 1][j - 1]);
					}

					// apply forces within the block
					Physics::apply_within_block(grid[i][j]);
				}
			}

			for (int i = 0; i < common::ParticleData::num_particles; i++) {
				double old_x = particles[i].x;
				double old_y = particles[i].y;
				common::move(particles[i], 0.005f);
				Physics::check_move(grid, &particles[i], old_x, old_y, m_BlocksPerSide);
			}			
		}


		//glm::mat4 testMat = glm::mat4(1.0f);
		//float* flatMat = static_cast<float*>(glm::value_ptr(testMat));

		//float* matrices[16] = new float[common::ParticleData::num_particles * 16];


		

	/*	float flatArr[16];
		memcpy(flatArr, flatMat, sizeof(float) * 16);

		spdlog::info("--------------------------");
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				std::cout << flatArr[j + i * 4];
			}
			std::cout << '\n';
		}
		spdlog::info("--------------------------");*/

	}

	void InstancedBinnedCPU::Draw(float deltaTime)
	{
		
		Novaura::Renderer::SetClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		Novaura::Renderer::Clear();
		Novaura::Renderer::BeginSceneInstanced(m_CameraController->GetCamera());	
		m_Gui->BeginFrame();

		float width = Novaura::InputHandler::GetCurrentWindow()->Width;
		float height = Novaura::InputHandler::GetCurrentWindow()->Height;
		float aspectRatio = Novaura::InputHandler::GetCurrentWindow()->AspectRatio;		
		
		for (int i = 0; i < common::ParticleData::num_particles; i++)
		{
			Novaura::Renderer::DrawInstancedCircle_glm(glm::vec3(particles[i].x -1.5f, particles[i].y-0.75, 0), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
		}	

		//Novaura::BatchRenderer::EndScene();
		Novaura::Renderer::EndInstancedCircles_glm();
		m_Gui->DrawStateButtons(m_StateInfo, particleScale);
		
		m_Gui->EndFrame();
	}

	

	void InstancedBinnedCPU::OnExit()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		spdlog::info(__FUNCTION__);
		free(particles);
		pvec::FreeGrid(grid, m_BlocksPerSide);
		Novaura::Renderer::ShutdownInstancedCircles_glm();
	}

	void InstancedBinnedCPU::Pause()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		//m_StateMachine->PushState(std::make_unique<PauseMenu>(m_Window, m_CameraController, m_StateMachine));
		
	}

	void InstancedBinnedCPU::Resume()
	{
		//Novaura::InputHandler::SetCurrentController(m_InputController);
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}

}
