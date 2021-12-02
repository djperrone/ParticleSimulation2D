#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace Math{

	struct Vector3f
	{
		union
		{
			float vec[3];
			float x, y, z;
		};
	};

	struct Vector4f
	{
		union
		{
			float vec[4];
			float x, y, z, w;
		};
	};

	struct FlatMatrix
	{
		union
		{
			float mat[16];
			Vector4f rows[4];
		};

		FlatMatrix()
		{
			memset(mat, 0, 16 * sizeof(float));
		}
	};

	void MakeIdentity_glm(FlatMatrix* dest);

	void MakeTranslation_glm(FlatMatrix* dest, const glm::vec3& vec);
	void MakeScale_glm(FlatMatrix* dest, const glm::vec3& vec);

	void MakeIdentity(FlatMatrix* dest);
	void MakeTranslation(FlatMatrix* dest, const Vector3f& vec);
	void MakeScale(FlatMatrix* dest, const Vector3f& vec);


	void TestIdentity();
	void TestTranslation();
	void TestScale();


}
