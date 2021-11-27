#include <sapch.h>
#include "PVec.h"

namespace pvec {

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "PVec.h"

#define DEFAULT_INIT_VEC(v) v.capacity = 0, v.pcount = 0, v.data = NULL 

    void PushParticle(ParticleVec& vec, common::particle_t* p)
    {
        //printf("push\n");
        if (vec.data == NULL || vec.capacity == 0)
        {
            vec.data = (common::particle_t**)malloc(8 * sizeof(common::particle_t*));
            vec.capacity = 8;
            vec.pcount = 0;
        }

        if (vec.pcount == vec.capacity)
        {
            common::particle_t** tmp = (common::particle_t**)malloc(vec.capacity * 2 * sizeof(common::particle_t*));
            memcpy(tmp, vec.data, vec.capacity * sizeof(common::particle_t**));
            free(vec.data);
            vec.data = tmp;
            vec.capacity *= 2;
        }
        vec.data[vec.pcount++] = p;
    }


    void EraseParticle(ParticleVec& vec, int index)
    {
        //printf("erase\n");

        if (vec.data == NULL || vec.capacity == 0)
        {
            printf("Can't erase element - vector not initialized");
        }

        if (index >= vec.capacity)
        {
            printf("Can't erase element - out of range");
            return;
        }
        if (vec.data[index] == NULL)
        {
            printf("Can't erase element - is NULL");
            return;
        }

        while (vec.data[index] != NULL && index + 1 < vec.capacity)
        {
            // printf("index %d, capacity %d ", index, vec.capacity);
            // printf("%f, %f, \n", vec.data[index]->x, vec.data[index]->y);
            // printf("%f, %f, \n", vec.data[index + 1]->x, vec.data[index + 1]->y);

            vec.data[index] = vec.data[index + 1];
            index++;
        }

        vec.data[--vec.pcount] = NULL;
    }


    ParticleVec** InitGrid3d(int length)
    {
        //printf("init\n");

        ParticleVec** particleVec = (ParticleVec**)malloc(length * sizeof(ParticleVec*));
        for (int i = 0; i < length; i++)
        {
            particleVec[i] = (ParticleVec*)malloc(length * sizeof(ParticleVec));
            for (int j = 0; j < length; j++)
            {
                DEFAULT_INIT_VEC(particleVec[i][j]);
            }
        }
        return particleVec;
    }


    void PrintVec(ParticleVec** grid, int blocks_per_side)
    {
        for (int i = 0; i < blocks_per_side; i++)
        {
            for (int j = 0; j < blocks_per_side; j++)
            {
                printf("%d, %d ", i, j);
                for (int k = 0; k < grid[i][j].pcount; k++)
                {
                    printf("%d: %f, %f, ", k, grid[i][j].data[k]->x, grid[i][j].data[k]->y);
                }
                printf("\n");
            }
        }
    }
}
