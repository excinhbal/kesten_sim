#ifndef MPI_KESTEN_H
#define MPI_KESTEN_H

#include <mpi.h>
#include "kestensimulation.h"

MPI_Datatype register_structural_events_type();

struct MpiInfo
{
    int world_size = 0;
    int rank = 0;
    int i_start = -1;
    int i_end = -1;
    MPI_Datatype MPI_Type_StructuralPlasticityEvent = MPI_DATATYPE_NULL;
};

template<typename P, typename L>
class MpiKestenSim : public KestenSimulation<P, L> {
    using KestenSimulation<P, L>::w;
    using KestenSimulation<P, L>::structual_events;
    using KestenSimulation<P, L>::t_begin;
    using KestenSimulation<P, L>::steps;

public:
    MpiKestenSim(const Parameters& p, const MpiInfo& mpiInfo);

    void mpiSendAndCollectWeights();
    void mpiSendAndCollectStrctEvents();
    void mpiSaveResults();

protected:
    int synchronizeActive(int n_active) override;

private:
    const MpiInfo mpiInfo;
    std::vector<double> w_all;
    std::vector<StructuralPlasticityEvent> structual_events_all;
};

std::ostream& operator<<(std::ostream& ostream, const MpiInfo& info);

#endif // MPI_KESTEN_H
