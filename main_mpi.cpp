#include "mpikesten.h"
#include "kestensimulation.h"
#include "main_shared.h"

int main(int argc, char** argv) {

    Parameters p = params_from_arguments(argc, argv);

    MPI_Init(nullptr, nullptr);

    MpiInfo mpiInfo;
    mpiInfo.MPI_Type_StructuralPlasticityEvent = register_structural_events_type();
    MPI_Comm_size(MPI_COMM_WORLD, &mpiInfo.world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiInfo.rank);

    mpiInfo.i_start = (mpiInfo.rank) * p.N_e / mpiInfo.world_size;
    mpiInfo.i_end = (mpiInfo.rank + 1) * p.N_e / mpiInfo.world_size;

    std::cout << mpiInfo << std::endl;

    {
        MpiKestenSim<Parameters, KestenStep> sim(p, mpiInfo);
        while (sim.hasNextStep()) {
            sim.doStep();
        }
        sim.mpiSendAndCollectWeights();
        sim.mpiSendAndCollectStrctEvents();
        if (mpiInfo.rank == 0) {
            sim.mpiSaveResults();
        }
    } // delete MpiKestenSim on non-root nodes the moment they are done

    MPI_Finalize();

    return 0;
}