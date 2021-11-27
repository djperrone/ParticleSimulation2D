#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "time_of_day.h"
//#include <sys/time.h>
#include "common.h"
#include <cstdlib>
#include "Novaura/Random.h"

namespace common {

    float ParticleData::density = 0.005f;
    float ParticleData::mass = 0.01f;
    float ParticleData::cutoff = 0.01f;
    float ParticleData::min_r = (cutoff / 100);
    float ParticleData::dt = 0.0005f;
    int ParticleData::num_particles = 50;
    double ParticleData::size = 0.0;


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
        ParticleData::size = sqrt(ParticleData::density * n);
    }

    //
    //  Initialize the particle positions and velocities
    //
    void init_particles(int n, particle_t* p)
    {


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
            int j = static_cast<int>(Novaura::Random::Uint32(0, 100000)) % (n - i);

            int k = shuffle[j];
            shuffle[j] = shuffle[n - i - 1];

            //
            //  distribute particles evenly to ensure proper spacing
            //
            p[i].x = ParticleData::size * (1. + (k % sx)) / (1 + sx);
            p[i].y = ParticleData::size * (1. + (k / sx)) / (1 + sy);

            //
            //  assign random velocities within a bound
            //
            p[i].vx = Novaura::Random::Float(0.0f, 1.0f) * 2 - 1;
            p[i].vy = Novaura::Random::Float(0.0f, 1.0f) * 2 - 1;
        }
        free(shuffle);
    }

    //
    //  interact two particles
    //
    void apply_force(particle_t& particle, particle_t& neighbor)
    {

        double dx = neighbor.x - particle.x;
        double dy = neighbor.y - particle.y;
        double r2 = dx * dx + dy * dy;
        if (r2 > ParticleData::cutoff * ParticleData::cutoff)
            return;
        r2 = fmax(r2, ParticleData::min_r * ParticleData::min_r);
        double r = sqrt(r2);

        //
        //  very simple short-range repulsive force
        //
        double coef = (1 - ParticleData::cutoff / r) / r2 / ParticleData::mass;
        particle.ax += coef * dx;
        particle.ay += coef * dy;
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
        p.vx += p.ax * ParticleData::dt;
        p.vy += p.ay * ParticleData::dt;
        p.x += p.vx * ParticleData::dt;
        p.y += p.vy * ParticleData::dt;

        //
        //  bounce from walls
        //
        while (p.x < 0 || p.x > ParticleData::size)
        {
            p.x = p.x < 0 ? -p.x : 2 * ParticleData::size - p.x;
            p.vx = -p.vx;
        }
        while (p.y < 0 || p.y > ParticleData::size)
        {
            p.y = p.y < 0 ? -p.y : 2 * ParticleData::size - p.y;
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
            fprintf(f, "%d %g\n", n, ParticleData::size);
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
