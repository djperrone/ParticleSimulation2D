#pragma once
namespace common {

	struct ParticleData
	{
		static float density;// = 0.03f;
		static float mass;// = 0.7f;
		static float cutoff; //= 0.095f;
		static float min_r; //= (cutoff / 100);
		static float dt; //= 0.0005f;
		static int num_particles;// = 5;
		static double size;
		static float scale;
	};
}
