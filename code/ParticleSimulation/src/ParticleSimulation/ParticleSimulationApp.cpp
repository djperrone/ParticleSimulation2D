#include "sapch.h"
#include "ParticleSimulationApp.h"

#include "States/BinnedGPU.h"
#include "States/BinnedGPU2.h"
#include "States/BinnedCPU.h"
#include "States/InstancedBinnedCPU.h"
#include "States/InstancedBinnedGPU.h"
#include "States/InstancedBinnedGPU_final.h"
#include "States/InstancedBinnedGPU_glm.h"
#include "States/SerialCPU.h"
#include "States/SerialCPU_final.h"
#include "States/StableFluidsGPU_test.h"
#include "States/StableFluidsGPU_test2.h"

namespace ParticleSimulation {
	ParticleSimulationApp::ParticleSimulationApp()
	{
		
	}
	ParticleSimulationApp::ParticleSimulationApp(std::string_view title, float width, float height)
		:Application(title, width, height)
	{		
		//m_StateMachine->PushState(std::make_unique<SerialCPU>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<BinnedCPU>(GetWindow(), m_CameraController, m_StateMachine));
	//	m_StateMachine->PushState(std::make_unique<InstancedBinnedCPU>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<InstancedBinnedGPU>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<StableFluidsGPU_test>(GetWindow(), m_CameraController, m_StateMachine));
		m_StateMachine->PushState(std::make_unique<StableFluidsGPU_test2>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<InstancedBinnedGPU_final>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<SerialCPU_final>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<InstancedBinnedGPU_glm>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<BinnedGPU>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<BinnedGPU2>(GetWindow(), m_CameraController, m_StateMachine));
	}
}