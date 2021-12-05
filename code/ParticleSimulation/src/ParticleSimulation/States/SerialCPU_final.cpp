#include "sapch.h"
#include "SerialCPU_final.h"

#include "Novaura/Novaura.h"
#include "Novaura/Collision/Collision.h"

#include "Novaura/Core/Application.h"

#include "CudaSrc/Physics.cuh"

#include "Novaura/Random.h"

namespace ParticleSimulation {
	

	SerialCPU_final::SerialCPU_final()
	{
		m_Window = Novaura::InputHandler::GetCurrentWindow();
		m_CameraController = Novaura::Application::GetCameraController();
		m_StateMachine = Novaura::Application::GetStateMachine();
		OnEnter();
	}
	
	SerialCPU_final::SerialCPU_final(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
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

	void SerialCPU_final::OnEnter()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		
		//Novaura::InputHandler::GetCurrentController().BindActionInputEvent(GLFW_PRESS, GLFW_KEY_ESCAPE, &SerialCPU_final::Pause, this);
		//Novaura::InputHandler::GetCurrentController().BindActionInputEvent(GLFW_PRESS, GLFW_KEY_ESCAPE, &SerialCPU_final::Pause, this);

		
		n_per_side = (int)sqrt(n);
		
		sq = StableFluids::FluidSquareCreate(n_per_side, d, v, dt);

		StableFluids::FluidSquareAddDensity(sq, n_per_side / 2, n_per_side / 2, 50);
		StableFluids::FluidSquareAddVelocity(sq, n_per_side / 2, n_per_side / 2, 3, 3);
		
		
		//particles = (common::particle_t*)malloc(common::ParticleData::num_particles * sizeof(common::particle_t));
		//common::set_size(common::ParticleData::num_particles);
		//common::init_particles(common::ParticleData::num_particles, particles);

		//particles = (common::particle_t*)malloc(common::ParticleData::num_particles * sizeof(common::particle_t));
		//common::set_size(common::ParticleData::num_particles);

		//common::init_particles(common::ParticleData::num_particles, particles);

	
	

	

		m_StateInfo.PAUSE = true;
		m_StateInfo.PLAY = false;
		m_StateInfo.RESET = false;
	}

	void SerialCPU_final::HandleInput()
	{
	}

	void SerialCPU_final::Update(float deltaTime)
	{		
		if (m_StateInfo.RESET)
		{
			//m_StateMachine->ReplaceCurrentState(std::make_unique<ParticleSimulation::SerialCPU_final>());
			OnExit();
			OnEnter();
		}
		m_CameraController->Update(*Novaura::InputHandler::GetCurrentWindow(), deltaTime);
		if (!m_StateInfo.PAUSE)
		{
			/*for (int i = 0; i < common::ParticleData::num_particles; i++)
			{
				particles[i].ax = particles[i].ay = 0;
				for (int j = 0; j < common::ParticleData::num_particles; j++)
					common::apply_force(particles[i], particles[j]);
			}*/

			//
			//  move particles
			//
			/*for (int i = 0; i < common::ParticleData::num_particles; i++)
				common::move(particles[i], deltaTime);*/

			FluidSquareStep(sq);
			

			//StableFluids::FluidSquareAddDensity(sq, n_per_side / 2, n_per_side / 2, 50);
			//StableFluids::FluidSquareAddVelocity(sq, n_per_side / 2, n_per_side / 2, 3, 3);
			
		}
		

	}

	void SerialCPU_final::Draw(float deltaTime)
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
		//Novaura::BatchRenderer::DrawRectangle(glm::vec3(-0.65f, 0.1f, 0.0f), glm::vec3(common::ParticleData::size, common::ParticleData::size, 1.0), glm::vec4(0.2f, 0.2f, 0.8f, 1.0f));
		//for (int i = 0; i < n; i++)
		//{
		//	//Novaura::BatchRenderer::DrawCircle(glm::vec3(particles[i].x - 1.5f, particles[i].y - 0.75, 0), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
		//	Novaura::BatchRenderer::DrawCircle(glm::vec3(sq->Vx[i] , sq->Vy[i], 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
		//}
		for (int i = 0; i < n_per_side; i++)
		{
			for (int j = 0; j < n_per_side; j++)
			{
				int scale = n / 150;
				float x = i * scale / width;
				float y = j * scale / height;
				int N = n_per_side;
				float d = sq->density[IX(i, j)];
				/*float addDensity = Novaura::Random::Float(0.0f, 25.f);
				float addVelocity = Novaura::Random::Float(-0.2f, 0.2f);*/
				/*float addDensity = Novaura::Random::Float(0.0f, 25.f);
				float addVelocityx = glm::sin(glfwGetTime()) * 0.2f;
				float addVelocityy = -glm::cos(glfwGetTime()) *0.2f;*/

				//float addDensity = Novaura::Random::Float(0.0f, 1000.f);
				float addDensity =5.0f;
				float addVelocityx = glm::sin(glfwGetTime()) * Novaura::Random::Float(-0.2f, 0.2f);
				float addVelocityy = glm::sin(glfwGetTime());// *Novaura::Random::Float(-0.2f, 0.2f);
				if (i == n_per_side / 2 && j == n_per_side / 2)
				{
					StableFluids::FluidSquareAddDensity(sq, i, j, addDensity);
					StableFluids::FluidSquareAddVelocity(sq, i, j, addVelocityx, addVelocityy);
					

				}
				//Novaura::BatchRenderer::DrawCircle(glm::vec3(particles[i].x - 1.5f, particles[i].y - 0.75, 0), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
					//Novaura::BatchRenderer::DrawCircle(glm::vec3(sq->Vx[i + j * n_per_side], sq->Vy[i + j * n_per_side], 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
				Novaura::BatchRenderer::DrawRectangle(glm::vec3(x, y, 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.0f,0.0f, glm::clamp(d,0.2f,1.0f), 100.0f/255.0f), glm::vec2(1.0f, 1.0f));
			}
			// cool
			//float addDensity = Novaura::Random::Float(0.0f, 25.f);
			//float addVelocityx = glm::sin(glfwGetTime()) * Novaura::Random::Float(-0.2f, 0.2f);
			//float addVelocityy = glm::sin(glfwGetTime());// *Novaura::Random::Float(-0.2f, 0.2f);
		}

		//for (int i = 0; i < n_per_side; i++)
		//{
		//	for (int j = 0; j < n_per_side; j++)
		//	{
		//		int scale = n / 75;
		//		float x = i * scale / width;
		//		float y = j * scale / height;
		//		int N = n_per_side;
		//		float d = sq->density[IX(i, j)];
		//		float addDensity = Novaura::Random::Float(0.0f, 25.f);
		//		float addVelocity = glm::sin(glfwGetTime());
		//		if (i == n_per_side / 2 && j == n_per_side / 2)
		//		{
		//			//StableFluids::FluidSquareAddDensity(sq, i, j, addDensity);
		//			//StableFluids::FluidSquareAddVelocity(sq, i, j, addVelocity, 0);

		//		}
		//		//Novaura::BatchRenderer::DrawCircle(glm::vec3(particles[i].x - 1.5f, particles[i].y - 0.75, 0), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
		//			//Novaura::BatchRenderer::DrawCircle(glm::vec3(sq->Vx[i + j * n_per_side], sq->Vy[i + j * n_per_side], 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), glm::vec2(1.0f, 1.0f));
		//		Novaura::BatchRenderer::DrawCircle(glm::vec3(x+0.5f, y+0.5f, 0.0f), glm::vec3(particleScale, particleScale, 0), glm::vec4(d, d, d, 1.0f), glm::vec2(1.0f, 1.0f));
		//	}


		//}

		
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

	

	void SerialCPU_final::OnExit()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		//free(particles);
		StableFluids::FluidSquareFree(sq);
	}

	void SerialCPU_final::Pause()
	{
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		//m_StateMachine->PushState(std::make_unique<PauseMenu>(m_Window, m_CameraController, m_StateMachine));
		
	}

	void SerialCPU_final::Resume()
	{
		//Novaura::InputHandler::SetCurrentController(m_InputController);
		//glfwSetInputMode(Novaura::InputHandler::GetCurrentWindow()->Window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}

}
