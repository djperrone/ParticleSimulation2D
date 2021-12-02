#include "sapch.h"
#include "Matrix.h"
#include <spdlog/spdlog.h>


namespace Math {

	void MakeIdentity_glm(FlatMatrix* dest)
	{
		glm::mat4 identity = glm::mat4(1.0f);
		memcpy(dest->mat, glm::value_ptr(identity), sizeof(float) * 16);
	}
	void MakeTranslation_glm(FlatMatrix* dest, const glm::vec3& vec)
	{
		glm::mat4 source = glm::translate(glm::mat4(1.0f), vec);
		memcpy(dest->mat, glm::value_ptr(source), sizeof(float) * 16);
	}
	void MakeScale_glm(FlatMatrix* dest, const glm::vec3& vec)
	{
		glm::mat4 source = glm::scale(glm::mat4(1.0f), vec);
		memcpy(dest->mat, glm::value_ptr(source), sizeof(float) * 16);
	}
	void MakeIdentity(FlatMatrix* dest)
	{
		dest->rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });
		dest->rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });
		dest->rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });
		dest->rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });
	}
	void MakeTranslation(FlatMatrix* dest, const Vector3f& vec)
	{
		dest->rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });
		dest->rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });
		dest->rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });
		dest->rows[3] = Vector4f({ vec.x, vec.y, vec.z, 1.0f });
	}
	void MakeScale(FlatMatrix* dest, const Vector3f& vec)
	{
		dest->rows[0] = Vector4f({ vec.x, 0.0f, 0.0f, 0.0f });
		dest->rows[1] = Vector4f({ 0.0f, vec.y, 0.0f, 0.0f });
		dest->rows[2] = Vector4f({ 0.0f, 0.0f,vec.z, 0.0f });
		dest->rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });
	}
	void TestIdentity()
	{
		spdlog::info(__FUNCTION__);

		glm::mat4 baseCase = glm::mat4(1.0f);
		FlatMatrix myMat;

		myMat.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });
		myMat.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });
		myMat.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });
		myMat.rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (myMat.mat[j + i * 4] != baseCase[i][j])
				{
					//std::cout << myMat.mat[j + i * 4] >> '\n';
					spdlog::error("Identity not correct");
					spdlog::info("mine:{:03.2f, glm: {:03.2f", myMat.mat[j + i * 4], (int)baseCase[i][j]);
					spdlog::info("mine: {}, glm: {}", 5, 6);
				}
			}
			
		}

		
	}
	void TestTranslation()
	{
		spdlog::info(__FUNCTION__);

		glm::mat4 baseCase = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f,3.0f,4.0f));
		FlatMatrix myMat;

		myMat.rows[0] = Vector4f({ 1.0f, 0.0f, 0.0f, 0.0f });
		myMat.rows[1] = Vector4f({ 0.0f, 1.0f, 0.0f, 0.0f });
		myMat.rows[2] = Vector4f({ 0.0f, 0.0f, 1.0f, 0.0f });
		myMat.rows[3] = Vector4f({ 2.0f, 3.0f, 4.0f, 1.0f });

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (myMat.mat[j + i * 4] != baseCase[i][j])
				{
					//std::cout << myMat.mat[j + i * 4] >> '\n';
					spdlog::error("translate not correct");
				}
			}

		}
	}
	void TestScale()
	{
		spdlog::info(__FUNCTION__);
		glm::mat4 baseCase = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f));
		FlatMatrix myMat;

		myMat.rows[0] = Vector4f({ 2.0f, 0.0f, 0.0f, 0.0f });
		myMat.rows[1] = Vector4f({ 0.0f, 3.0f, 0.0f, 0.0f });
		myMat.rows[2] = Vector4f({ 0.0f, 0.0f, 4.0f, 0.0f });
		myMat.rows[3] = Vector4f({ 0.0f, 0.0f, 0.0f, 1.0f });

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				if (myMat.mat[j + i * 4] != baseCase[i][j])
				{
					//std::cout << myMat.mat[j + i * 4] >> '\n';
					spdlog::error("scale not correct");
				}
			}

		}
	}
}
