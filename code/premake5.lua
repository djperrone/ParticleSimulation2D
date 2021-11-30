workspace "ParticleSimulation"
	architecture "x64"
	startproject "ParticleSimulation"

	configurations
	{
		"Debug",
		"Release"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

-- include directories relative to root folder (solution directory)

IncludeDir = {}
IncludeDir["GLFW"] = "ParticleSimulation/vendor/GLFW/include"
IncludeDir["Glad"] = "ParticleSimulation/vendor/Glad/include"
IncludeDir["ImGui"] = "ParticleSimulation/vendor/imgui"
IncludeDir["stb_image"] = "ParticleSimulation/vendor/stb_image"
IncludeDir["glm"] = "ParticleSimulation/vendor/glm"
IncludeDir["FreeType"] = "ParticleSimulation/vendor/FreeType/include"

group "Dependencies"
include "ParticleSimulation/vendor/GLFW"
include "ParticleSimulation/vendor/Glad"
include "ParticleSimulation/vendor/FreeType"
include "ParticleSimulation/vendor/imgui"
group ""

project "ParticleSimulation"
	location "ParticleSimulation"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"
	buildcustomizations "BuildCustomizations/CUDA 11.5"
	

	targetdir ("bin/" .. outputdir .. "/%{prj.name}")
	objdir ("bin-int/" .. outputdir .. "/%{prj.name}")	

	pchheader "sapch.h"
	pchsource "ParticleSimulation/src/sapch.cpp"

	defines
	{
		"_CRT_SECURE_NO_WARNINGS"	
	}

	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/vendor/stb_image/**.h",
		"%{prj.name}/vendor/stb_image/**.cpp",	
		"%{prj.name}/vendor/glm/glm/**.hpp",
		"%{prj.name}/vendor/glm/glm/**.inl"
	}

	

	includedirs
	{
		"%{prj.name}/src",		
		"%{prj.name}/vendor/spdlog/include",				
		"%{IncludeDir.FreeType}",				
		"%{IncludeDir.GLFW}",
		"%{IncludeDir.Glad}",
		"%{IncludeDir.ImGui}",	
		"%{IncludeDir.stb_image}",	
		"%{IncludeDir.glm}"
	}
	libdirs
	{
		--"%{prj.name}/vendor/imgui/bin"
	}

	links
	{
		"GLFW",
		"Glad",		
		"FreeType",
		 "ImGui",
		--"ImGui.lib",
		"opengl32.lib"
	}

	filter "system:windows"
		systemversion "latest"

		defines
		{			
			"GLFW_INCLUDE_NONE"		
		}

	filter "configurations:Debug"			
		runtime "Debug"
		symbols "on"

	filter "configurations:Release"
		runtime "Release"
		optimize "on"
