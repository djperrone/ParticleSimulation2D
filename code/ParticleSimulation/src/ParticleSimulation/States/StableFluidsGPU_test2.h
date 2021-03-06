#pragma once
#include "Novaura/StateMachine/State.h"
#include "Novaura/Camera/CameraController.h"
//#include "Novaura/Primitives/Rectangle.h"
#include "StateInfo.h"


#include "common/common.h"
#include "../gui.h"
#include "common/Block.h"
#include "common/PVec.h"

#include "FinalCode/fluid.h"
#include "FinalCode/utilities.h"

#include "CudaSrc/FinalCuda/Fluid.cuh"


namespace ParticleSimulation {


	class StableFluidsGPU_test2 : public Novaura::State
	{
	public:
		StableFluidsGPU_test2();		
		StableFluidsGPU_test2(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine);
		
		virtual void OnEnter() override;

		virtual void HandleInput() override;
		virtual void Update(float deltaTime)override;
		virtual void Draw(float deltaTime) override;

		virtual void OnExit() override;

		virtual void Pause() override;
		virtual void Resume() override;

		void Debug();
	
	private:		

		double m_CurrentTime = 0.0;
		double m_PreviousTime = 0.0;
		float particleScale = 0.03f;

		StateInfo m_StateInfo;	
	

		std::unique_ptr<Pgui::Gui> m_Gui;

		StableFluidsCuda::FluidSquare sq;
		StableFluidsCuda::FluidSquare sq_cpu;
		StableFluids::FluidSquare* sq_test;


		int n = 100;
		float d = 0;
		float v = .00001;
		float dt = .005;
		int n_per_side;
	};
}
