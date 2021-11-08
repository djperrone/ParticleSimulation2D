#include "sapch.h"
#include "ParticleSimulationApp.h"

#include "States/Simulation.h"


namespace ParticleSimulation {
	ParticleSimulationApp::ParticleSimulationApp()
	{
		
	}
	ParticleSimulationApp::ParticleSimulationApp(std::string_view title, float width, float height)
		:Application(title, width, height)
	{		
		m_StateMachine->PushState(std::make_unique<Simulation>(GetWindow(), m_CameraController, m_StateMachine));		
	}
}