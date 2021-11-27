#include "sapch.h"
#include "Simulation.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "CudaSrc/Physics.cuh"



namespace ParticleSimulation {
	
	

	Simulation::Simulation()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}
	
	Simulation::Simulation(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
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

	void Simulation::OnEnter()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		
		//Novaura::InputHandler::GetCurrentController().BindActionInputEvent(GLFW_PRESS, GLFW_KEY_ESCAPE, &Simulation::Pause, this);
		//Novaura::InputHandler::GetCurrentController().BindActionInputEvent(GLFW_PRESS, GLFW_KEY_ESCAPE, &Simulation::Pause, this);

		
		

		
		
		//particles = (common::particle_t*)malloc(common::ParticleData::num_particles * sizeof(common::particle_t));
		//common::set_size(common::ParticleData::num_particles);
		//common::init_particles(common::ParticleData::num_particles, particles);

		particles = (common::particle_t*)malloc(common::ParticleData::num_particles * sizeof(common::particle_t));
		common::set_size(common::ParticleData::num_particles);

		common::init_particles(common::ParticleData::num_particles, particles);

		m_BlocksPerSide = (int)round(sqrt(common::ParticleData::num_particles));
		double block_size = common::ParticleData::size / m_BlocksPerSide;
		if (block_size < .01) {
			m_BlocksPerSide = (int)common::ParticleData::size / common::ParticleData::cutoff;
			block_size = common::ParticleData::size / m_BlocksPerSide;
		}
		
		//m_Grid = (common::Block*)malloc(m_BlocksPerSide * m_BlocksPerSide * sizeof(common::Block));
		//for (int i = 0; i < m_BlocksPerSide * m_BlocksPerSide; i++) {
		//	m_Grid[i].pcount = 0;
		//}

		//// fill the grid
		//for (int i = 0; i < common::ParticleData::num_particles; i++) {
		//	int block_x = (int)(particles[i].x / block_size);
		//	int block_y = (int)(particles[i].y / block_size);
		//	common::push_particle(m_Grid[block_x * m_BlocksPerSide + block_y], &particles[i], i);
		//}

		grid = pvec::InitGrid3d(m_BlocksPerSide);
		// fill the grid
		for (int i = 0; i < common::ParticleData::num_particles; i++) {
			int block_x = (int)particles[i].x / block_size;
			int block_y = (int)particles[i].y / block_size;

			//grid[block_x][block_y].push_back(&particles[i]);
			pvec::PushParticle(grid[block_x][block_y], (&particles[i]));
		}

		//cudaMalloc((void**)&m_Grid_gpu, blocks_per_side * blocks_per_side * sizeof(common::Block));
		//Physics::InitParticles(particles, d_particles);




		//particles[0].x = 1.5;
		//particles[1].x = -1.5;
		//particles[3].y = 0.8;
		//particles[4].y =- 0.8;
		/*for (int i = 0; i < common::ParticleData::num_particles; i++)
		{
			std::cout << particles[i].x << ' ' << particles[i].y << '\n';
		}*/

		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;
	}

	void Simulation::HandleInput()
	{
	}

	void Simulation::Update(float deltaTime)
	{		
		if (m_StateInfo.RESET)
		{
			//m_StateMachine->ReplaceCurrentState(std::make_unique<ParticleSimulation::Simulation>());
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
		

	}

	void Simulation::Draw(float deltaTime)
	{
		
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

	

	void Simulation::OnExit()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		free(particles);
	}

	void Simulation::Pause()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		//m_StateMachine->PushState(std::make_unique<PauseMenu>(m_Window, m_CameraController, m_StateMachine));
		
	}

	void Simulation::Resume()
	{
		//Novaura::InputHandler::SetCurrentController(m_InputController);
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}

}
