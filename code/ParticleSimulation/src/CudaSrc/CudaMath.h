#pragma once
#include "Math/Matrix.h"

namespace CudaMath {
	void MakeIdentity_cpu(Math::Matrix44f* dest);

	void MakeTranslation_cpu(Math::Matrix44f* dest, const glm::vec3& vec);
	void MakeScale_cpu(Math::Matrix44f* dest, const glm::vec3& vec);

	void TestIdentity_cpu();
	void TestTranslation_cpu();
	void TestScale_cpu();

}
