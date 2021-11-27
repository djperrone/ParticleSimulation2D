#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <cstdio>
#include "particle_t.h"
#include "ParticleData.h"

#define density1 0.005
#define mass1    0.01
#define cutoff1  0.01
#define min_r1   (cutoff1/100)
#define dt1      0.0005

namespace common {


	inline int mymin(int a, int b) { return a < b ? a : b; }
	inline int mymax(int a, int b) { return a > b ? a : b; }

//#define density 0.0005
//#define mass    0.01
//#define cutoff  0.01
//#define min_r   (cutoff/100)
//#define dt      0.0005
	//float dt = 0.005;
	//struct ParticleData
	//{
	//	static float density;// = 0.03f;
	//	static float mass;// = 0.7f;
	//	static float cutoff; //= 0.095f;
	//	static float min_r; //= (cutoff / 100);
	//	static float dt; //= 0.0005f;
	//	static int num_particles;// = 5;
	//	static double size;
	//};

	//__global__ float density;// = 0.03f;
	//__global__ float mass;// = 0.7f;
	//__global__ float cutoff; //= 0.095f;
	//__global__ float min_r; //= (cutoff / 100);
	//__global__ float dt; //= 0.0005f;
	//__global__ int num_particles;// = 5;
	//__global__ double size;

	struct PDataGPU
	{
		
	};

	//
	//  saving parameters
	//
	const int NSTEPS = 1000;
	const int SAVEFREQ = 10;

	//
	// particle data structure
	//
	/*typedef struct
	{
		double x;
		double y;
		double vx;
		double vy;
		double ax;
		double ay;

	} particle_t;*/

	//
	//  timing routines
	//
	double read_timer();

	//
	//  simulation routines
	//

	void set_size(int n);
	void set_size_for_gpu(int n);
	void init_particles(int n, particle_t* p);
	void init_particles_for_gpu(int n, particle_t* p);
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
#endif
