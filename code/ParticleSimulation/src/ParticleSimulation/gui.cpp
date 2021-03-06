#include "sapch.h"
#include "Gui.h"
#include <imgui.h>
#include <imconfig.h>
#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include "../common/common.h"

#include "Novaura/Novaura.h"

namespace Pgui {

    Gui::Gui(std::shared_ptr<Novaura::Window>& window)
        : m_Window(window)
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

        //io.ConfigFlags |= ImGuiConfigFlags_ViewPortsNoTaskBarIcons; //???
        //io.ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge; //???

        // Setup ImGui style
        ImGui::StyleColorsDark();

        // When viewports are enabled we tweak WindowWounding/WindowBg so platform windows can look identical to regualr ones.
        ImGuiStyle& style = ImGui::GetStyle();

        // Setup platform renderer bindings

        ImGui_ImplGlfw_InitForOpenGL(m_Window->Window, true);
        ImGui_ImplOpenGL3_Init("#version 410");
    }


    Gui::~Gui()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void Gui::Draw()
    {
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);




        static float f = 0.0f;
        static int counter = 0;
        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
        float tx = 0.0f;
        ImGui::InputFloat("test tx", &tx, 0.5f, 2, 0);// ImGui::SameLine(150);
       // int n = 0;
        //ImGui::SliderInt("test n", &n, 0, 10'000);

       // ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
       // ImGui::Checkbox("Another Window", &show_another_window);


       // ImGui::SliderFloat("v_mod", &common::velocity_modifier, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
       // ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

       // if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
         //   counter++;
       // ImGui::SameLine();
        //ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }

    void Gui::DrawStateButtons(ParticleSimulation::StateInfo& stateInfo, float& pscale)
    {
        int save_num = common::ParticleData::num_particles;
        int save_density = common::ParticleData::density;
        int save_mass = common::ParticleData::mass;
        int save_cutoff = common::ParticleData::cutoff;

        ImGui::Begin("Particle Simulation");
        if (!m_Changed)
        {
            if (stateInfo.PAUSE)
            {
                if (ImGui::Button("PLAY"))
                {
                    stateInfo.PAUSE = !stateInfo.PAUSE;
                }

            }
            else
            {
                if (ImGui::Button("PAUSE"))
                {
                    stateInfo.PAUSE = !stateInfo.PAUSE;
                }
            }
        }



        if (ImGui::Button("RESET"))
        {
            stateInfo.RESET = !stateInfo.RESET;
            if (m_Changed == true)m_Changed = false;
        }
        ImGui::Separator();

        if (stateInfo.PAUSE)
        {

            ImGui::Separator();
            ImGui::SliderInt("num particles", &common::ParticleData::num_particles, 1, 10'000);
            ImGui::SliderFloat("density", &common::ParticleData::density, 0.0005f, 0.05f);
            ImGui::SliderFloat("mass", &common::ParticleData::mass, 0.0005f, 5.0f);
            ImGui::SliderFloat("cutoff", &common::ParticleData::cutoff, 0.0005f, 5.0f);
            //ImGui::SliderFloat("scale", &pscale, 0.0005f, 5.0f);
            ImGui::InputFloat("scale", &pscale, 0.0005f, 1.5f, 5);

            /*if (save_num != common::ParticleData::num_particles || save_density != common::ParticleData::density ||
                save_mass != common::ParticleData::mass || save_cutoff != common::ParticleData::cutoff) {
                m_Changed = true;
                spdlog::info("changed!");
            }*/

            if (save_num != common::ParticleData::num_particles) { stateInfo.RESET = true; m_Changed = false;
            }
        }

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        ImGui::End();
    }

    void Gui::DrawDockSpace(ParticleSimulation::StateInfo& stateInfo)
    {
        static bool dockspaceOpen = true;
        static bool opt_fullscreen = true;
        static bool opt_padding = false;
        static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

        // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
        // because it would be confusing to have two docking targets within each others.
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        if (opt_fullscreen)
        {
            ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->GetWorkPos());
            ImGui::SetNextWindowSize(viewport->GetWorkSize());
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        }
        else
        {
            dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
        }

        // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
        // and handle the pass-thru hole, so we ask Begin() to not render a background.
        if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
            window_flags |= ImGuiWindowFlags_NoBackground;

        // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
        // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
        // all active windows docked into it will lose their parent and become undocked.
        // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
        // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
        if (!opt_padding)
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("DockSpace", &dockspaceOpen, window_flags);
        if (!opt_padding)
            ImGui::PopStyleVar();

        if (opt_fullscreen)
            ImGui::PopStyleVar(2);

        // DockSpace
        ImGuiIO& io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
        {
            ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
        }


        //if (ImGui::BeginMenuBar())
        //{
        //    if (ImGui::BeginMenu("File"))
        //    {
        //        // Disabling fullscreen would allow the window to be moved to the front of other windows,
        //        // which we can't undo at the moment without finer window depth/z control.              

        //        if (ImGui::MenuItem("Exit"))
        //        {

        //        }
        //      
        //      
        //        ImGui::EndMenu();
        //    }         

        //    ImGui::EndMenuBar();
        //}

        //DrawStateButtons(stateInfo);


        ImGui::End();
    }



    void Gui::BeginFrame()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void Gui::EndFrame()
    {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(m_Window->Width, m_Window->Height);


        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
    }

}