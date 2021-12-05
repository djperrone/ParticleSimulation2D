#include "sapch.h"
#include "Renderer.h"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include "Novaura/Renderer/TextureLoader.h"
#include "Novaura/Renderer/Vertex.h"
#include "Novaura/Renderer/IndexBuffer.h"
#include "Novaura/Renderer/VertexArray.h"
#include "Novaura/Renderer/TextLoader.h"

#include "Novaura/Renderer/VertexBuffer.h"

#include "Novaura/CudaGLInterop/Interop.h"
#include "common/particle_t.h"
#include "CudaSrc/Physics.cuh"

#include "Novaura/CudaGLInterop/helper_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cudagl.h>



namespace Novaura {

	struct RenderData
	{	
		std::unique_ptr<VertexArray> s_VertexArray;
		std::unique_ptr<IndexBuffer> s_IndexBuffer;
		std::unique_ptr<VertexBuffer> s_VertexBuffer;

		std::unique_ptr<Shader> TextureShader;
		std::unique_ptr<Shader> ColorShader;

		glm::vec4 DefaultRectangleVertices[4];
		glm::vec2 DefaultTextureCoords[4];

		std::unique_ptr<Shader> TextRenderShader;		

		std::unique_ptr<Shader> InstancedCircleShader;	

		glm::mat4* ModelMatrices_glm;

		CudaMath::Matrix44f* ModelMatrices;
		
		cudaGraphicsResource_t positionsVBO_CUDA = 0;
		CudaMath::Matrix44f* translationMatrices_d = nullptr;
		CudaMath::Matrix44f* scaleMatrix_d = nullptr;
	
		unsigned int MaxCircles;
		const unsigned int InstancedIndexCount = 6;
		unsigned int CircleCounter = 0;

		GLuint instanceVBO;	
		unsigned int sphereVAO;
		unsigned int vbo, ebo;
	};	

	static RenderData s_RenderData;

	void Renderer::SetClearColor(float r, float g, float b, float a)
	{		
		glClearColor(r, g, b, 1.0f);
	}	

	void Renderer::Clear()
	{
		glClear(GL_COLOR_BUFFER_BIT);
	}

	void Renderer::Init()
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);

		//glEnable(GL_DEPTH_TEST);
		//glDepthFunc(GL_LESS);
		glEnable(GL_STENCIL_TEST);
		glStencilMask(0x00); // disable writing to the stencil buffer
		glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
		glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

		SetClearColor(0.05f, 0.05f, 0.05f, 1.0f);

		s_RenderData.s_VertexArray = std::make_unique<VertexArray>();
		s_RenderData.s_VertexBuffer = std::make_unique<VertexBuffer>();
		s_RenderData.TextureShader = std::make_unique<Shader>("Assets/Shaders/TextureShader.glsl");
		s_RenderData.ColorShader = std::make_unique<Shader>("Assets/Shaders/BasicColorShader.glsl");
		s_RenderData.TextRenderShader = std::make_unique<Shader>("Assets/Shaders/TextRenderShader.glsl");

		constexpr unsigned int numIndices = 6;
		unsigned int indices[numIndices] = {
			0,1,2,
			2,3,0		
		};		

		s_RenderData.s_IndexBuffer = std::make_unique <IndexBuffer>(indices, numIndices);
		 // aspect ratio
		s_RenderData.DefaultRectangleVertices[0] = glm::vec4(-0.5f, -0.5f, 0.0f, 1.0f);
		s_RenderData.DefaultRectangleVertices[1] = glm::vec4( 0.5f, -0.5f, 0.0f, 1.0f);
		s_RenderData.DefaultRectangleVertices[2] = glm::vec4( 0.5f,  0.5f, 0.0f, 1.0f);
		s_RenderData.DefaultRectangleVertices[3] = glm::vec4(-0.5f,  0.5f, 0.0f, 1.0f);

		s_RenderData.DefaultTextureCoords[0] = glm::vec2(0.0f, 0.0f);
		s_RenderData.DefaultTextureCoords[1] = glm::vec2(1.0f, 0.0f);
		s_RenderData.DefaultTextureCoords[2] = glm::vec2(1.0f, 1.0f);
		s_RenderData.DefaultTextureCoords[3] = glm::vec2(0.0f, 1.0f);

		s_RenderData.InstancedCircleShader = std::make_unique<Shader>("Assets/Shaders/InstancedCircleShader.glsl");
		glGenVertexArrays(1, &s_RenderData.sphereVAO);
		glGenBuffers(1, &s_RenderData.vbo);
		glGenBuffers(1, &s_RenderData.ebo);
		glGenBuffers(1, &s_RenderData.instanceVBO);

	}

	void Renderer::Init(const Camera& camera)
	{
	}

	

	void Renderer::UpdateInstancedCircleMatrices_glm(unsigned int amount)
	{
		
	}
	
	void Renderer::BeginScene(Shader& shader, const Camera& camera)
	{		
		shader.Bind();		
		shader.SetUniformMat4f("u_ViewProjectionMatrix", camera.GetViewProjectionMatrix());
	}

	void Renderer::BeginScene(const Camera& camera)
	{
		s_RenderData.ColorShader->Bind();
		//s_RenderData.ColorShader->SetUniformMat4f("u_ViewProjectionMatrix", camera.GetViewProjectionMatrix());
		s_RenderData.ColorShader->SetUniformMat4f("u_ViewMatrix", camera.GetViewMatrix());
		s_RenderData.ColorShader->SetUniformMat4f("u_ProjectionMatrix", camera.GetProjectionMatrix());

		s_RenderData.TextureShader->Bind();		
		s_RenderData.TextureShader->SetUniformMat4f("u_ViewProjectionMatrix", camera.GetViewProjectionMatrix());
		//s_RenderData.TextureShader->SetUniformMat4f("u_ProjectionMatrix", camera.GetProjectionMatrix());

		s_RenderData.TextRenderShader->Bind();
		//s_RenderData.TextRenderShader->SetUniformMat4f("u_ViewMatrix", camera.GetViewMatrix());
		s_RenderData.TextRenderShader->SetUniformMat4f("u_ViewProjectionMatrix", camera.GetViewProjectionMatrix());
	}



	void Renderer::DrawRectangle(const Rectangle& rectangle, const glm::vec2& quantity)
	{		
		DrawRectangle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetColor(), quantity);		
	}


	void Renderer::DrawRectangle(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, const glm::vec2& quantity)
	{
		s_RenderData.ColorShader->Bind();	
		s_RenderData.ColorShader->SetUniform2f("u_Quantity", quantity.x, quantity.y);

		std::vector<VertexData> vertices;
		vertices.reserve(4);

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), scale);

		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.DefaultTextureCoords[0]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.DefaultTextureCoords[1]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.DefaultTextureCoords[2]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.DefaultTextureCoords[3]);

		s_RenderData.s_VertexBuffer->SetData(vertices);
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), 0);
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, Color));
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, TexCoord));

		s_RenderData.s_VertexArray->Bind();
		s_RenderData.s_VertexBuffer->Bind();
		s_RenderData.s_IndexBuffer->Bind();

		//shader.SetUniform4f("u_Color", m_Color);
		glDrawElements(GL_TRIANGLES, s_RenderData.s_IndexBuffer->GetCount(), GL_UNSIGNED_INT, nullptr);
		s_RenderData.TextureShader->Bind();
	}

	void Renderer::DrawRectangle(const Rectangle& rectangle, std::string_view texture, const glm::vec2& quantity)
	{
		DrawRectangle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetColor(), texture, quantity);
	}

	void Renderer::DrawRectangle(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, std::string_view texture, const glm::vec2& quantity)
	{
		Texture tex = TextureLoader::LoadTexture(texture);
		tex.Bind();
		s_RenderData.TextureShader->SetUniform2f("u_Quantity", quantity.x, quantity.y);

		std::vector<VertexData> vertices;
		vertices.reserve(4);

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), scale);

		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.DefaultTextureCoords[0]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.DefaultTextureCoords[1]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.DefaultTextureCoords[2]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.DefaultTextureCoords[3]);

		s_RenderData.s_VertexBuffer->SetData(vertices);
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), 0);
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, Color));
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, TexCoord));

		s_RenderData.s_VertexArray->Bind();
		s_RenderData.s_VertexBuffer->Bind();
		s_RenderData.s_IndexBuffer->Bind();

		//shader.SetUniform4f("u_Color", m_Color);
		glDrawElements(GL_TRIANGLES, s_RenderData.s_IndexBuffer->GetCount(), GL_UNSIGNED_INT, nullptr);
		tex.UnBind();
	}

	void Renderer::DrawRotatedRectangle(const Rectangle& rectangle, const glm::vec2& quantity)
	{		
		DrawRotatedRectangle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetRotation(), rectangle.GetColor(), quantity);		
	}

	void Renderer::DrawRotatedRectangle(const glm::vec3& position, const glm::vec3& scale, float rotation, const glm::vec4& color, const glm::vec2& quantity)
	{
		std::vector<VertexData> vertices;
		vertices.reserve(4);
		s_RenderData.TextureShader->SetUniform2f("u_Quantity", quantity.x, quantity.y);

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position)
			* glm::rotate(glm::mat4(1.0f), glm::radians(rotation), glm::vec3(0.0f, 0.0f, 1.0f))
			* glm::scale(glm::mat4(1.0f), scale);


		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.DefaultTextureCoords[0]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.DefaultTextureCoords[1]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.DefaultTextureCoords[2]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.DefaultTextureCoords[3]);

		s_RenderData.s_VertexBuffer->SetData(vertices);
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), 0);
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, Color));
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, TexCoord));

		s_RenderData.s_VertexArray->Bind();
		s_RenderData.s_VertexBuffer->Bind();
		s_RenderData.s_IndexBuffer->Bind();

		//shader.SetUniform4f("u_Color", m_Color);
		glDrawElements(GL_TRIANGLES, s_RenderData.s_IndexBuffer->GetCount(), GL_UNSIGNED_INT, nullptr);
	}

	void Renderer::DrawRotatedRectangle(const Rectangle& rectangle, std::string_view texture, const glm::vec2& quantity)
	{
		DrawRotatedRectangle(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetRotation(), rectangle.GetColor(), texture, quantity);
	}

	void Renderer::DrawRotatedRectangle(const glm::vec3& position, const glm::vec3& scale, float rotation, const glm::vec4& color, std::string_view texture, const glm::vec2& quantity)
	{
		Texture tex = TextureLoader::LoadTexture(texture);
		s_RenderData.TextureShader->SetUniform2f("u_Quantity", quantity.x, quantity.y);


		tex.Bind();

		std::vector<VertexData> vertices;
		vertices.reserve(4);

		glm::mat4 transform = glm::translate(glm::mat4(1.0f), position)
			* glm::rotate(glm::mat4(1.0f), glm::radians(rotation), glm::vec3(0.0f, 0.0f, 1.0f))
			* glm::scale(glm::mat4(1.0f), scale);


		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[0], color, s_RenderData.DefaultTextureCoords[0]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[1], color, s_RenderData.DefaultTextureCoords[1]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[2], color, s_RenderData.DefaultTextureCoords[2]);
		vertices.emplace_back(transform * s_RenderData.DefaultRectangleVertices[3], color, s_RenderData.DefaultTextureCoords[3]);

		s_RenderData.s_VertexBuffer->SetData(vertices);
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), 0);
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, Color));
		s_RenderData.s_VertexArray->AddBuffer(*s_RenderData.s_VertexBuffer, 2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, TexCoord));
					 
		s_RenderData.s_VertexArray->Bind();
		s_RenderData.s_VertexBuffer->Bind();
		s_RenderData.s_IndexBuffer->Bind();

		//shader.SetUniform4f("u_Color", m_Color);
		glDrawElements(GL_TRIANGLES, s_RenderData.s_IndexBuffer->GetCount(), GL_UNSIGNED_INT, nullptr);
		tex.UnBind();
	}

	void Renderer::RenderText(std::string text, float x, float y, float scale, glm::vec3 color)
	{

		//spdlog::info(__FUNCTION__);
		s_RenderData.TextRenderShader->Bind();
		//s_RenderData.TextRenderShader->SetUniform3f("textColor", color);

		unsigned int VAO, VBO;
		std::unique_ptr<VertexArray> textVA;
		std::unique_ptr<VertexBuffer> textBuffer;
		//textVA->AddBuffer(*textBuffer,0,4, GL_FLOAT, GL_FALSE, 4,0);
		//s_RenderData.VertexArray->AddBuffer(*s_RenderData.VertexBuffer, 1, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), offsetof(VertexData, Color));

		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		glUniform3f(glGetUniformLocation(s_RenderData.TextRenderShader->GetID(), "textColor"), color.x, color.y, color.z);
		glActiveTexture(GL_TEXTURE0);
		glBindVertexArray(VAO);

		// iterate through all characters
		std::vector<std::vector<float>> verts;

		std::string::const_iterator c;
		float startingXpos = 0.0f, endXPos = 0.0f;
		bool first = false;
		int count = text.length();
		for (c = text.begin(); c != text.end(); c++)
		{
			
			Text::Character ch = TextLoader::LoadedCharacters[*c];

			float xpos = x + ch.Bearing.x * scale;
			float ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

			float w = ch.Size.x * scale;
			float h = ch.Size.y * scale;
			// update VBO for each character

			if (!first)
			{
				first = true;
				startingXpos = x + ch.Bearing.x;
			}

			if (count-- == 0)
			{
				endXPos = x + ch.Bearing.x;
			}
			

			float vertices[6][4] = {
				{ xpos,     ypos + h,   0.0f, 0.0f },
				{ xpos,     ypos,       0.0f, 1.0f },
				{ xpos + w, ypos,       1.0f, 1.0f },

				{ xpos,     ypos + h,   0.0f, 0.0f },
				{ xpos + w, ypos,       1.0f, 1.0f },
				{ xpos + w, ypos + h,   1.0f, 0.0f }
			};
			// render glyph texture over quad
			glBindTexture(GL_TEXTURE_2D, ch.TextureID);
			// update content of VBO memory
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			// render quad
			glDrawArrays(GL_TRIANGLES, 0, 6);
			verts.clear();

			// now advance cursors for next glyph (note that advance is number of 1/64 pixels)
			x += (ch.Advance >> 6) * scale; // bitshift by 6 to get value in pixels (2^6 = 64)
		}
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);

		spdlog::info("{0}, {1}", startingXpos, endXPos);
	}


	

	void Renderer::InitInstancedCircles_glm(unsigned int amount, float scale, const glm::vec4& color)
	{
		constexpr unsigned int numIndices = 6;

		unsigned int indices[numIndices] = {
			0,1,2,
			2,3,0
		};

		std::vector<InstancedVertexData_glm> vertices;
		vertices.reserve(4);	
	
		vertices.emplace_back(glm::vec4(-0.5f, -0.5f, 0.0f, scale), color);
		vertices.emplace_back(glm::vec4(0.5f, -0.5f, 0.0f, scale), color);
		vertices.emplace_back(glm::vec4(0.5f, 0.5f, 0.0f, scale), color);
		vertices.emplace_back(glm::vec4(-0.5f, 0.5f, 0.0f, scale), color);

		glBindVertexArray(s_RenderData.sphereVAO);
		s_RenderData.MaxCircles = amount;

		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(InstancedVertexData_glm), &vertices[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_RenderData.ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(InstancedVertexData_glm), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(InstancedVertexData_glm), (void*)offsetof(InstancedVertexData_glm, Color));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_RenderData.ebo);	

		s_RenderData.ModelMatrices_glm = new glm::mat4[s_RenderData.MaxCircles];

		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.instanceVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * amount, &s_RenderData.ModelMatrices_glm[0], GL_DYNAMIC_DRAW);
		glBindVertexArray(s_RenderData.sphereVAO);		

		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4)));
		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));
		glEnableVertexAttribArray(5);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));

		glVertexAttribDivisor(2, 1);
		glVertexAttribDivisor(3, 1);
		glVertexAttribDivisor(4, 1);
		glVertexAttribDivisor(5, 1);	

		//CudaGLInterop::RegisterCudaGLBuffer(s_RenderData.positionsVBO_CUDA, s_RenderData.instanceVBO);
	}	

	void Renderer::DrawInstancedCircle_glm(const Rectangle& rectangle, const glm::vec2& quantity)
	{
		DrawInstancedCircle_glm(rectangle.GetPosition(), rectangle.GetScale(), rectangle.GetColor(), quantity);
	}
	
	void Renderer::DrawInstancedCircle_glm(const glm::vec3& position, const glm::vec3& scale, const glm::vec4& color, const glm::vec2& quantity)
	{
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, position) * glm::scale(glm::mat4(1.0f), scale);		
		s_RenderData.ModelMatrices_glm[s_RenderData.CircleCounter++] = model;
	}

	void Renderer::BeginSceneInstanced(const Camera& camera)
	{
		s_RenderData.InstancedCircleShader->Bind();
		s_RenderData.InstancedCircleShader->SetUniformMat4f("u_ViewProjectionMatrix", camera.GetViewProjectionMatrix());
	}

	void Renderer::EndInstancedCircles_glm()
	{
		glBindVertexArray(s_RenderData.sphereVAO);

		//if (s_RenderData.CircleCounter != s_RenderData.MaxCircles) spdlog::info(__FUNCTION__,"not enough circles?...");		

		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.instanceVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * s_RenderData.MaxCircles, &s_RenderData.ModelMatrices_glm[0], GL_DYNAMIC_DRAW);
		glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, s_RenderData.CircleCounter);		

		s_RenderData.CircleCounter = 0;
	}

	
	
	void Renderer::ShutdownInstancedCircles_glm()
	{
		delete[] s_RenderData.ModelMatrices_glm;
		//delete[] s_RenderData.ModelMatrices;
	}
	void Renderer::ShutdownInstancedCircles()
	{
		//delete[] s_RenderData.ModelMatrices;
		cudaFree(s_RenderData.scaleMatrix_d);
		cudaFree(s_RenderData.translationMatrices_d);
	
	}


	void Renderer::InitInteropInstancedCircles_glm(unsigned int amount, float scale, const glm::vec4& color)
	{
		constexpr unsigned int numIndices = 6;

		unsigned int indices[numIndices] = {
			0,1,2,
			2,3,0
		};

		std::vector<InstancedVertexData_glm> vertices;
		vertices.reserve(4);

		vertices.emplace_back(glm::vec4(-0.5f, -0.5f, 0.0f, scale), color);
		vertices.emplace_back(glm::vec4(0.5f, -0.5f, 0.0f, scale), color);
		vertices.emplace_back(glm::vec4(0.5f, 0.5f, 0.0f, scale), color);
		vertices.emplace_back(glm::vec4(-0.5f, 0.5f, 0.0f, scale), color);

		glBindVertexArray(s_RenderData.sphereVAO);
		s_RenderData.MaxCircles = amount;

		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(InstancedVertexData_glm), &vertices[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_RenderData.ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(InstancedVertexData_glm), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(InstancedVertexData_glm), (void*)offsetof(InstancedVertexData_glm, Color));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_RenderData.ebo);

		s_RenderData.ModelMatrices_glm = new glm::mat4[s_RenderData.MaxCircles];

		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.instanceVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * amount, 0, GL_DYNAMIC_DRAW);
		//CudaGLInterop::RegisterCudaGLBuffer(s_RenderData.positionsVBO_CUDA, &s_RenderData.instanceVBO);
		cudaGraphicsGLRegisterBuffer(&s_RenderData.positionsVBO_CUDA, s_RenderData.instanceVBO, cudaGraphicsRegisterFlagsWriteDiscard);

		//glBindVertexArray(s_RenderData.sphereVAO);

		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4)));
		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));
		glEnableVertexAttribArray(5);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));

		glVertexAttribDivisor(2, 1);
		glVertexAttribDivisor(3, 1);
		glVertexAttribDivisor(4, 1);
		glVertexAttribDivisor(5, 1);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//glBindBuffer(GL_VERTEX_ARRAY,0);
		//CudaGLInterop::InitDevices();
	}

	void Renderer::OnReset(unsigned int amount)
	{
		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.instanceVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * amount, 0, GL_DYNAMIC_DRAW);
		//CudaGLInterop::RegisterCudaGLBuffer(s_RenderData.positionsVBO_CUDA, &s_RenderData.instanceVBO);
		cudaGraphicsGLRegisterBuffer(&s_RenderData.positionsVBO_CUDA, s_RenderData.instanceVBO, cudaGraphicsRegisterFlagsWriteDiscard);
	}

	void Renderer::RegisterCudaGLDevice()
	{
		cudaGraphicsGLRegisterBuffer(&s_RenderData.positionsVBO_CUDA, s_RenderData.instanceVBO, cudaGraphicsRegisterFlagsWriteDiscard);
	}

	void Renderer::EndInteropInstancedCircles()
	{		
		glBindVertexArray(s_RenderData.sphereVAO);	
		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.instanceVBO);	
		glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, s_RenderData.MaxCircles);
		glBindBuffer(GL_ARRAY_BUFFER, 0);		
	}

	void Renderer::UpdateMatricesInterop_glm(common::particle_t* particles_gpu, int num_particles)
	{
		size_t num_bytes;
		glm::mat4* matrices = nullptr;		

		cudaGraphicsMapResources(1, &s_RenderData.positionsVBO_CUDA, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&matrices, &num_bytes, s_RenderData.positionsVBO_CUDA);
		if (num_bytes != s_RenderData.MaxCircles * sizeof(glm::mat4))
			spdlog::info("bytes dont match updatematriucesinterop");

		Physics::UpdateMatrices_cpu_glm(matrices, particles_gpu, num_particles);
		cudaGraphicsUnmapResources(1, &s_RenderData.positionsVBO_CUDA, 0);
		
	}
	void Renderer::InitInteropInstancedCircles(common::particle_t* particles_gpu, unsigned int amount, float scale, const CudaMath::Vector4f& color)
	{
		constexpr unsigned int numIndices = 6;

		unsigned int indices[numIndices] = {
			0,1,2,
			2,3,0
		};

		std::vector<InstancedVertexData> vertices;
		vertices.reserve(4);

		vertices.emplace_back(CudaMath::Vector4f{-0.5f, -0.5f, 0.0f, scale}, color);
		vertices.emplace_back(CudaMath::Vector4f{ 0.5f, -0.5f, 0.0f, scale}, color);
		vertices.emplace_back(CudaMath::Vector4f{ 0.5f,  0.5f, 0.0f, scale}, color);
		vertices.emplace_back(CudaMath::Vector4f{-0.5f,  0.5f, 0.0f, scale}, color);		

		glBindVertexArray(s_RenderData.sphereVAO);
		s_RenderData.MaxCircles = amount;

		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(InstancedVertexData), &vertices[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_RenderData.ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(InstancedVertexData), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(InstancedVertexData), (void*)offsetof(InstancedVertexData, Color));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_RenderData.ebo);		

		glBindBuffer(GL_ARRAY_BUFFER, s_RenderData.instanceVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(CudaMath::Matrix44f) * amount, 0, GL_DYNAMIC_DRAW);
	
		cudaGraphicsGLRegisterBuffer(&s_RenderData.positionsVBO_CUDA, s_RenderData.instanceVBO, cudaGraphicsRegisterFlagsWriteDiscard);	

		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(CudaMath::Matrix44f), (void*)0);
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(CudaMath::Matrix44f), (void*)(sizeof(CudaMath::Vector4f)));
		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(CudaMath::Matrix44f), (void*)(2 * sizeof(CudaMath::Vector4f)));
		glEnableVertexAttribArray(5);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(CudaMath::Matrix44f), (void*)(3 * sizeof(CudaMath::Vector4f)));

		glVertexAttribDivisor(2, 1);
		glVertexAttribDivisor(3, 1);
		glVertexAttribDivisor(4, 1);
		glVertexAttribDivisor(5, 1);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);		

		CudaMath::Matrix44f scaleMatrix;

		MAKE_SCALE(scaleMatrix, scale);
		cudaMalloc((void**)&s_RenderData.scaleMatrix_d, sizeof(CudaMath::Matrix44f));
		cudaMemcpy(s_RenderData.scaleMatrix_d, &scaleMatrix, sizeof(CudaMath::Matrix44f), cudaMemcpyHostToDevice);	
		cudaMalloc((void**)&s_RenderData.translationMatrices_d, sizeof(CudaMath::Matrix44f) * amount);
	}
	
	void Renderer::UpdateMatricesInterop(common::particle_t* particles_gpu, float scale, int num_particles)
	{	
		if (num_particles != s_RenderData.MaxCircles) spdlog::error(__FUNCTION__,' ', __LINE__);

		size_t num_bytes;
		CudaMath::Matrix44f* resultMatrices = nullptr;
		cudaGraphicsMapResources(1, &s_RenderData.positionsVBO_CUDA, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&resultMatrices, &num_bytes, s_RenderData.positionsVBO_CUDA);
		if (num_bytes != s_RenderData.MaxCircles * sizeof(CudaMath::Matrix44f))
			spdlog::info("bytes dont match updatematriucesinterop");
		
		CudaMath::MakeTranslationMatrices_cpu(s_RenderData.translationMatrices_d, particles_gpu, num_particles);

		CudaMath::MatMul44Batch_cpu(s_RenderData.translationMatrices_d, s_RenderData.scaleMatrix_d, resultMatrices, num_particles);
		cudaGraphicsUnmapResources(1, &s_RenderData.positionsVBO_CUDA, 0);		
	}
}
