#include <mpi.h>
#include "kestensim.h"

struct MpiInfo
{
    int world_size = 0;
    int rank = 0;
    int i_start = -1;
    int i_end = -1;
};

std::ostream& operator<<(std::ostream& ostream, const MpiInfo& info)
{
    ostream << "MpiInfo { " << info.rank << " of " << info.world_size
            << " with range [" << info.i_start << ", " << info.i_end << ")"
            << " }";
    return ostream;
}

void mpikesten(const Parameters& p)
{
    MPI_Init(nullptr, nullptr);

    MpiInfo mpiInfo;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiInfo.world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiInfo.rank);

    mpiInfo.i_start = (mpiInfo.rank) * p.N_e / mpiInfo.world_size;
    mpiInfo.i_end = (mpiInfo.rank + 1) * p.N_e / mpiInfo.world_size;

    std::cout << mpiInfo << std::endl;

    MPI_Finalize();
}