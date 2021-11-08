#include "sapch.h"
#include "BatchTest.h"

#include <glm/glm.hpp>
#include "Novaura/Renderer/Renderer.h"
#include "Novaura/Renderer/BatchRenderer.h"
#include <spdlog/spdlog.h>

#include "Novaura/Renderer/Texture.h"
#include "Novaura/Renderer/TextureLoader.h"
#include "Novaura/Renderer/TextLoader.h"
#include "Novaura/Input/InputHandler.h"

namespace ParticleSimulation {	

	BatchTest::BatchTest(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine)
	{
		m_Window = window;
		m_CameraController = cameraController;
		m_StateMachine = stateMachine;
		OnEnter();
	}

	void BatchTest::OnEnter()
	{

		Novaura::BatchRenderer::Init();
		Novaura::Renderer::Init();
		Novaura::TextLoader::Init();

		m_Rect = std::make_unique<Novaura::Rectangle>(glm::vec2(0.0f, 0.0f), glm::vec2(0.5f, 0.5f), glm::vec4(0.2f, 0.2f, 0.8f, 1.0f));
		Novaura::InputHandler::GetCurrentController().BindAxisInputEvent(GLFW_KEY_T, []() {spdlog::info("test axis event T"); });
		//Novaura::Texture tex = Novaura::TextureLoader::LoadTexture("Assets/Textures/Spaceship.png");
		/*Novaura::InputHandler::GetCurrentController().BindActionInputEvent(GLFW_PRESS, GLFW_MOUSE_BUTTON_1, [m_Rect = &m_Rect, m_Window = &m_Window]()
			{
				if (Novaura::InputHandler::IsRectHovered(*m_Rect->get()))
				{
					spdlog::info("collision deteced!");
				}

			});*/


	}


	void BatchTest::HandleInput()
	{
	}
	namespace batchTest {
		float velocity = 0.01f;
	}
	void BatchTest::Update(float deltaTime)
	{
		m_CameraController->Update(GetWindow(), deltaTime);		
		float aspectRatio = m_Window->Width / m_Window->Height;		

		
		m_Rect->GetPosition().x += batchTest::velocity;
		if (m_Rect->GetPosition().x > aspectRatio)
		{
			batchTest::velocity *= -1;
		}

		if (m_Rect->GetPosition().x < -aspectRatio)
		{
			batchTest::velocity *= -1;
		}


		//Draw(deltaTime);		

	}

	void BatchTest::Draw(float deltaTime)
	{
		Novaura::Renderer::SetClearColor(0.05f, 0.05f, 0.05f, 1.0f);
		Novaura::Renderer::Clear();
		//Novaura::Renderer::DrawRectangle(*m_Rect);
		Novaura::Renderer::BeginScene(m_CameraController->GetCamera());
		//Novaura::Renderer::DrawRotatedRectangle(*m_Rect, "Assets/Textures/Spaceship.png");		
		Novaura::Renderer::RenderText("A", 0.0F, 0.0F,1.0f, glm::vec3(1.0f, 0.0f, 0.0f));

		/*spdlog::info("{0:.5f}", glfwGetTime() * 10.0f);
		m_Rect->SetRotation(glfwGetTime() * 10.0f);
		Novaura::BatchRenderer::BeginScene(m_CameraController->GetCamera());
		Novaura::BatchRenderer::DrawRectangle(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 1.0f), glm::vec4(0.2f, 0.8f, 0.2f, 1.0f), "Assets/Textures/UFO.png");
		Novaura::BatchRenderer::DrawRectangle(glm::vec3(0.0f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.2f), glm::vec4(0.8f, 0.2f, 0.2f, 1.0f), "Assets/Textures/UFO.png");
		Novaura::BatchRenderer::DrawRectangle(glm::vec3(0.5f, 0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 1.0f), glm::vec4(0.5f, 0.8f, 0.8f, 1.0f));*/

	//	Novaura::BatchRenderer::DrawRectangle(glm::vec3(-0.50f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.2f), glm::vec4(0.5f, 0.5f, 0.8f, 1.0f), "Assets/Textures/Spaceship.png");
		//Novaura::BatchRenderer::DrawRectangle(glm::vec3(0.75f, 0.75f, 0.0f), glm::vec3(0.5f, 0.5f, 1.0f), glm::vec4(0.7f, 0.3f, 0.5f, 1.0f));
		//Novaura::BatchRenderer::RenderText("HELLO", 0.0F, 0.0F,1.0f, glm::vec3(1.0f, 0.0f, 0.0f));
		//Novaura::BatchRenderer::EndScene();
	}

	

	void BatchTest::OnExit()
	{
	}

	void BatchTest::Pause()
	{
	}

	void BatchTest::Resume()
	{
	}
}
