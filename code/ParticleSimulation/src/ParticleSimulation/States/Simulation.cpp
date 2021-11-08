#include "sapch.h"
#include "Simulation.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"





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

		
		


		
		particles = (common::particle_t*)malloc(common::ParticleData::num_particles * sizeof(common::particle_t));
		common::set_size(common::ParticleData::num_particles);
		init_particles(common::ParticleData::num_particles, particles);
		particles[0].x = 1.5;
		particles[1].x = -1.5;
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
			//m_ObjectManager->Update(deltaTime);

			simulation_time = common::read_timer();

			//for (int step = 0; step < common::NSTEPS; step++)
			//{
			navg = 0;
			davg = 0.0;
			dmin = 1.0;
			//
			//  compute forces
			//
			for (int i = 0; i < common::ParticleData::num_particles; i++)
			{
				particles[i].ax = particles[i].ay = 0;
				for (int j = 0; j < common::ParticleData::num_particles; j++)
					apply_force(particles[i], particles[j], &dmin, &davg, &navg);
			}

			//
			//  move particles
			//
			for (int i = 0; i < common::ParticleData::num_particles; i++)
				move(particles[i], deltaTime);

			if (navg) {
				absavg += davg / navg;
				nabsavg++;
			}
			if (dmin < absmin) absmin = dmin;
			//}
			simulation_time = common::read_timer() - simulation_time;
			/*	for (int i = 0; i < n; i++)
				{
					std::cout << particles[i].vx << ' ' << particles[i].vy << '\n';
				}*/

				/*for (int i = 0; i < n; i++)
				{
					while (particles[i].x < 0.1)
						particles[i].x *= 10;
				}*/
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
		float scale = common::ParticleData::density * 60.0f;
		//Novaura::BatchRenderer::StencilDraw(glm::vec3(0.5f, 0.5f, 0.0f), glm::vec3(0.25f, 0.25f, 0.0f), glm::vec4(0.5f, 0.1f, 0.8f, 1.0f), glm::vec4(0.1f, 0.8f, 0.1f, 1.0f));
		//Novaura::BatchRenderer::StencilDraw(glm::vec3(0.5f, 0.0f, 0.0f), glm::vec3(2.0f, 1.5f, 0.0f), glm::vec4(1.0f, 1.0f, 1.0f, 0.0f), glm::vec4(1.0f));
		Novaura::BatchRenderer::DrawRectangle(glm::vec3(-0.65f, 0.1f, 0.0f), glm::vec3(scale, scale, 1.0), glm::vec4(0.2f, 0.2f, 0.8f, 1.0f));
		for (int i =0; i < common::ParticleData::num_particles; i++)
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
