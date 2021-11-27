#include "sapch.h"
#include "ParticleSimulationApp.h"

#include "States/BinnedGPU.h"
#include "States/BinnedCPU.h"
#include "States/SerialCPU.h"


namespace ParticleSimulation {
	ParticleSimulationApp::ParticleSimulationApp()
	{
		
	}
	ParticleSimulationApp::ParticleSimulationApp(std::string_view title, float width, float height)
		:Application(title, width, height)
	{		
		//m_StateMachine->PushState(std::make_unique<SerialCPU>(GetWindow(), m_CameraController, m_StateMachine));
		//m_StateMachine->PushState(std::make_unique<BinnedCPU>(GetWindow(), m_CameraController, m_StateMachine));
		m_StateMachine->PushState(std::make_unique<BinnedGPU>(GetWindow(), m_CameraController, m_StateMachine));
	}
}