#include "Block.h"
#include "CudaSrc/BlockFunctions.cuh"

namespace common {
	void push_particle(common::Block& b, common::particle_t* particle, int idx)
	{
		b.particles[b.pcount] = particle;
		b.ids[b.pcount] = idx;
		b.pcount++;
	}
	void erase_particle(common::Block& b, int idx)
	{
		while (idx < b.pcount - 1) {
			b.particles[idx] = b.particles[idx + 1];
			b.ids[idx] = b.ids[idx + 1];
			idx++;
		}

		b.pcount--;
	}
}
