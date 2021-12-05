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

	struct Matrix44f
	{
		union
		{
			float mat[16];
			Vector4f rows[4];
		};

		Matrix44f()
		{
			memset(mat, 0, 16 * sizeof(float));
		}
	};

	void MakeIdentity_glm(Matrix44f* dest);

	void MakeTranslation_glm(Matrix44f* dest, const glm::vec3& vec);
	void MakeScale_glm(Matrix44f* dest, const glm::vec3& vec);

	void MakeIdentity(Matrix44f* dest);
	void MakeTranslation(Matrix44f* dest, const Vector3f& vec);
	void MakeScale(Matrix44f* dest, const Vector3f& vec);


	void TestIdentity();
	void TestTranslation();
	void TestScale();


}
