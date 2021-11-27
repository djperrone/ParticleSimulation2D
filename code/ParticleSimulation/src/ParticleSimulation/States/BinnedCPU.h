#pragma once
#include "Novaura/StateMachine/State.h"
#include "Novaura/Camera/CameraController.h"
//#include "Novaura/Primitives/Rectangle.h"
#include "StateInfo.h"


#include "common/common.h"
#include "../gui.h"
#include "common/Block.h"
#include "common/PVec.h"

namespace ParticleSimulation {


	class BinnedCPU : public Novaura::State
	{
	public:
		BinnedCPU();		
		BinnedCPU(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine);
		
		virtual void OnEnter() override;

		virtual void HandleInput() override;
		virtual void Update(float deltaTime)override;
		virtual void Draw(float deltaTime) override;

		virtual void OnExit() override;

		virtual void Pause() override;
		virtual void Resume() override;

	
	private:		

		double m_CurrentTime = 0.0;
		double m_PreviousTime = 0.0;
		float particleScale = 0.05f;

		int m_BlocksPerSide;

		StateInfo m_StateInfo;
		

		common::particle_t* particles;
		
		
		

		//int n;
		double simulation_time;
		int navg, nabsavg = 0;
		double davg, dmin, absmin = 1.0, absavg = 0.0;

		std::unique_ptr<Pgui::Gui> m_Gui;

		// old pvec
		pvec::ParticleVec** grid;




	};
}
