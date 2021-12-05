#pragma once
#include "Novaura/StateMachine/State.h"
#include "Novaura/Camera/CameraController.h"
//#include "Novaura/Primitives/Rectangle.h"
#include "StateInfo.h"


#include "common/common.h"
#include "../gui.h"
#include "common/Block.h"
#include "common/PVec.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



namespace ParticleSimulation {


	class InstancedBinnedGPU_final : public Novaura::State
	{
	public:
		InstancedBinnedGPU_final();		
		InstancedBinnedGPU_final(std::shared_ptr<Novaura::Window> window, std::shared_ptr<Novaura::CameraController> cameraController, std::shared_ptr<Novaura::StateMachine> stateMachine);
		
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
		float particleScale = 0.15f;

		common::particle_t* particles;
		common::particle_t* particles_gpu;		
		common::Block* grid_gpu;

		size_t blocks_per_side;
		double block_size;
	
		//int n;
		double simulation_time;
		int navg, nabsavg = 0;
		double davg, dmin, absmin = 1.0, absavg = 0.0;

		std::unique_ptr<Pgui::Gui> m_Gui;
		StateInfo m_StateInfo;

	};
}
