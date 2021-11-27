#ifndef COMMON_GPU
#define COMMON_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include "ParticleData.h"

namespace common_gpu {


	inline int mymin(int a, int b) { return a < b ? a : b; }
	inline int mymax(int a, int b) { return a > b ? a : b; }

	//#define density 0.0005
	//#define mass    0.01
	//#define cutoff  0.01
	//#define min_r   (cutoff/100)
	//#define dt      0.0005
	//float test1 = 0.5f;
	//__device__ const float test = 0.0f;
	struct ParticleDataGPU
	{
		float density;// = 0.03f;
		float mass;// = 0.7f;
		float cutoff; //= 0.095f;
		float min_r; //= (cutoff / 100);
		float dt; //= 0.0005f;
		int num_particles;// = 5;
		double size;

		void Init()
		{
			density = common::ParticleData::density;
			mass = common::ParticleData::mass;
			cutoff = common::ParticleData::cutoff;
			min_r = common::ParticleData::min_r;
			dt = common::ParticleData::dt;
			num_particles = common::ParticleData::num_particles;
			size = common::ParticleData::size;
		}
	};
	//__constant__ float density = 0.04f;
	//__constant__ float size = 0.0f;
	//zstatic ParticleData* pData;

	//__global__ float density;// = 0.03f;
	//__global__ float mass;// = 0.7f;
	//__global__ float cutoff; //= 0.095f;
	//__global__ float min_r; //= (cutoff / 100);
	//__global__ float dt; //= 0.0005f;
	//__global__ int num_particles;// = 5;
	//__global__ double size;

	

	//
	//  saving parameters
	//
	const int NSTEPS = 1000;
	const int SAVEFREQ = 10;

	//
	// particle data structure
	//
	typedef struct
	{
		double x;
		double y;
		double vx;
		double vy;
		double ax;
		double ay;

	} particle_t;

	//
	//  timing routines
	//
	double read_timer();

	//
	//  simulation routines
	//

	void set_size(int n);
	void init_particles(int n, particle_t* p);
	void apply_force(particle_t& particle, particle_t& neighbor);
	void move(particle_t& p, float deltaTime);

	//
	//  I/O routines
	//
	FILE* open_save(char* filename, int n);
	void save(FILE* f, int n, particle_t* p);

	//
	//  argument processing routines
	//
	int find_option(int argc, char** argv, const char* option);
	int read_int(int argc, char** argv, const char* option, int default_value);
	char* read_string(int argc, char** argv, const char* option, char* default_value);

}



#endif // !COMMON_GPU
