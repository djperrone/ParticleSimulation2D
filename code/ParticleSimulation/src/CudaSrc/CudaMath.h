#pragma once
#include "Math/Matrix.h"

namespace CudaMath {
	void MakeIdentity_cpu(Math::FlatMatrix* dest);

	void MakeTranslation_cpu(Math::FlatMatrix* dest, const glm::vec3& vec);
	void MakeScale_cpu(Math::FlatMatrix* dest, const glm::vec3& vec);

	void TestIdentity_cpu();
	void TestTranslation_cpu();
	void TestScale_cpu();

}
