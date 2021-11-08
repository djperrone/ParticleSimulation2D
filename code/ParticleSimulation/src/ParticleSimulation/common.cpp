#include "sapch.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctime>
//#include <sys/time.h>
#include "common.h"
#include "time_of_day.h"
#include <cstdlib>
#include <glm/glm.hpp>

#include "Novaura/Random.h"
namespace common {    
    
    float ParticleData::density = 0.03f;
    float ParticleData::mass    =   0.7f;
    float ParticleData::cutoff = 0.095f;
    float ParticleData::min_r =  (cutoff/100);
    float ParticleData::dt    =  0.0005f;
    int ParticleData::num_particles = 5;

    double size;

    //
    //  tuned constants
    //

//
//  timer
//
    double read_timer()
    {
        static bool initialized = false;
        static struct timeval start;
        struct timeval end;
        if (!initialized)
        {
            gettimeofday(&start, NULL);
            initialized = true;
        }
        gettimeofday(&end, NULL);
        return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    }

    //
    //  keep density constant
    //
    void set_size(int n)
    {
        size = sqrt(ParticleData::density * n);
    }

    //
    //  Initialize the particle positions and velocities
    //
    void init_particles(int n, particle_t* p)
    {
        //srand48(time(NULL));
        int sx = (int)ceil(sqrt((double)n));
        int sy = (n + sx - 1) / sx;

        int* shuffle = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++)
            shuffle[i] = i;

        for (int i = 0; i < n; i++)
        {
            //
            //  make sure particles are not spatially sorted
            //
            //int j = static_cast<int>(Novaura::Random::Uint32(0, (int)glm::pow(2, 31))) % (n - i);
            int j = static_cast<int>(Novaura::Random::Uint32(0, 100000)) % (n - i);

            //int j = lrand48() % (n - i);
            int k = shuffle[j];
            shuffle[j] = shuffle[n - i - 1];

            //
            //  distribute particles evenly to ensure proper spacing
            //
            p[i].x = size * (1. + (k % sx)) / (1 + sx);
            p[i].y = size * (1. + (k / sx)) / (1 + sy);

            //
            //  assign random velocities within a bound
            //
            p[i].vx = Novaura::Random::Float(0.0f, 1.0f) * 2 - 1;
            p[i].vy = Novaura::Random::Float(0.0f, 1.0f) * 2 - 1;

            // p[i].vx = drand48() * 2 - 1;
             //p[i].vy = drand48() * 2 - 1;
        }
        free(shuffle);
    }

    //
    //  interact two particles
    //
    void apply_force(particle_t& particle, particle_t& neighbor, double* dmin, double* davg, int* navg)
    {

        double dx = neighbor.x - particle.x;
        double dy = neighbor.y - particle.y;
        double r2 = dx * dx + dy * dy;
        if (r2 > ParticleData::cutoff * ParticleData::cutoff)
            return;
        if (r2 != 0)
        {
            if (r2 / (ParticleData::cutoff * ParticleData::cutoff) < *dmin * (*dmin))
                *dmin = sqrt(r2) / ParticleData::cutoff;
            (*davg) += sqrt(r2) / ParticleData::cutoff;
            (*navg)++;
        }

        r2 = fmax(r2, ParticleData::min_r * ParticleData::min_r);
        double r = sqrt(r2);



        //
        //  very simple short-range repulsive force
        //
        double coef = (1 - ParticleData::cutoff / r) / r2 / ParticleData::mass;
        particle.ax += coef * dx;
        particle.ay += coef * dy;
        //particle.ax *= 0.01;
        //particle.ay *= 0.01;
    }

    //
    //  integrate the ODE
    //
    void move(particle_t& p, float deltaTime)
    {
        //
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        //
        p.vx += p.ax;// * deltaTime * velocity_modifier;
        p.vy += p.ay;// * deltaTime * velocity_modifier;
       // p.vx *= 0.95;
       // p.vy *= 0.95;
        p.x += p.vx * deltaTime;
        p.y += p.vy * deltaTime;

        //
        //  bounce from walls
        //
        while (p.x < 0 || p.x > size)
        {
            p.x = p.x < 0 ? -p.x : 2 * size - p.x;
            p.vx = -p.vx;
        }
        while (p.y < 0 || p.y > size)
        {
            p.y = p.y < 0 ? -p.y : 2 * size - p.y;
            p.vy = -p.vy;
        }
    }

    //
    //  I/O routines
    //
    void save(FILE* f, int n, particle_t* p)
    {
        static bool first = true;
        if (first)
        {
            fprintf(f, "%d %g\n", n, size);
            first = false;
        }
        for (int i = 0; i < n; i++)
            fprintf(f, "%g %g\n", p[i].x, p[i].y);
    }

    //
    //  command line option processing
    //
    int find_option(int argc, char** argv, const char* option)
    {
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], option) == 0)
                return i;
        return -1;
    }

    int read_int(int argc, char** argv, const char* option, int default_value)
    {
        int iplace = find_option(argc, argv, option);
        if (iplace >= 0 && iplace < argc - 1)
            return atoi(argv[iplace + 1]);
        return default_value;
    }

    char* read_string(int argc, char** argv, const char* option, char* default_value)
    {
        int iplace = find_option(argc, argv, option);
        if (iplace >= 0 && iplace < argc - 1)
            return argv[iplace + 1];
        return default_value;
    }
}