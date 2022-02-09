#include "mpikesten.h"

MPI_Datatype register_structural_events_type()
{
    MPI_Datatype mpi_new_type;
    int blockLengths[] = {1, 1, 1, 1};
    MPI_Aint displacements[] = {
            offsetof(StructuralPlasticityEvent, type),
            offsetof(StructuralPlasticityEvent, t),
            offsetof(StructuralPlasticityEvent, i),
            offsetof(StructuralPlasticityEvent, j)
    };
    MPI_Datatype types[] = {MPI_INT, MPI_DOUBLE, MPI_INT, MPI_INT};
    MPI_Type_create_struct(
            4, blockLengths, displacements, types, &mpi_new_type
    );
    MPI_Type_commit(&mpi_new_type);
    return mpi_new_type;
}

template<typename P, typename L>
MpiKestenSim<P, L>::MpiKestenSim(const Parameters& p, const MpiInfo& mpiInfo_)
        : KestenSimulation<P, L>(p, NodeParameters{
            .N_e =  mpiInfo_.i_end-mpiInfo_.i_start,
            .neuronOffset =  mpiInfo_.i_start,
            .seedOffset =  mpiInfo_.rank,
        })
        , mpiInfo(mpiInfo_)
        , w_all(0)
{

}

template<typename P, typename L>
int MpiKestenSim<P, L>::synchronizeActive(int n_active)
{
    int n_active_all;
    MPI_Allreduce(&n_active, &n_active_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//    std::cout << mpiInfo.rank << " has active " << n_active << " and got all active " << n_active_all << std::endl;
    return n_active_all;
}

template<typename P, typename L>
void MpiKestenSim<P, L>::mpiSendAndCollectWeights()
{
    // TODO make w storage fully continuous
    auto size_acc = [](const int& acc, const auto& neuron_w) { return acc + neuron_w.size(); };
    int n_active = std::accumulate(w.cbegin(), w.cend(), 0, size_acc);
    std::vector<double> own_w(n_active, 0.0);
    auto own_w_inserter = own_w.begin();
    for (const auto& neuron_w: w) {
        for (const auto w_: neuron_w) {
            *own_w_inserter = w_;
            own_w_inserter++;
        }
    }

    std::vector<int> counts(0);
    if (mpiInfo.rank == 0) {
        counts.resize(mpiInfo.world_size);
        std::cout << "mpiInfo.world_size" << " " << mpiInfo.world_size << std::endl;
    }
    std::cout << "sending " << n_active << std::endl;
    MPI_Gather(&n_active, 1, MPI_INT,
               counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (mpiInfo.rank == 0) {
        for (const auto& c: counts) std::cout << c << " ";
        std::cout << std::endl;
    }

    int n_active_all = std::accumulate(counts.cbegin(), counts.cend(), 0);
    if (mpiInfo.rank == 0) { // root, we receive
        w_all.resize(n_active_all, 0.0);
        std::cout << "MPI(" << mpiInfo.rank << ")" << " receiving " << w_all.size() << std::endl;
    }
    std::vector<int> offsets(counts.size());
    std::partial_sum(counts.cbegin(), counts.cend(), offsets.begin());
    std::cout << "MPI(" << mpiInfo.rank << ")" << " sending " << own_w.size() << std::endl;
    MPI_Gatherv(own_w.data(), own_w.size(), MPI_DOUBLE,
                w_all.data(), offsets.data(), counts.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

template<typename P, typename L>
void MpiKestenSim<P, L>::mpiSendAndCollectStrctEvents()
{
    structual_events.reverse();
    auto eventBefore = [](const StructuralPlasticityEvent& ev1, const StructuralPlasticityEvent& ev2) { return ev1.t < ev2.t; };
    if (mpiInfo.rank == 0) { // root => receive
        std::vector<StructuralPlasticityEvent> own_source(structual_events.cbegin(), structual_events.cend());
        structual_events.clear();
        std::vector<StructuralPlasticityEvent> events_target{0};
        for (int i = 1; i < mpiInfo.world_size; ++i) {
            MPI_Status status;
            MPI_Probe(i, -1, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, mpiInfo.MPI_Type_StructuralPlasticityEvent, &count);
            std::vector<StructuralPlasticityEvent> events(count);
            events_target.resize(own_source.size() + count);
            MPI_Recv(events.data(), count, mpiInfo.MPI_Type_StructuralPlasticityEvent, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            std::merge(own_source.cbegin(), own_source.cend(), events.cbegin(), events.cend(), events_target.begin(), eventBefore);
            own_source = std::move(events_target);
            events_target = {};
        }
        structual_events_all = std::move(own_source);
    } else {
        std::vector<StructuralPlasticityEvent> own_data(structual_events.cbegin(), structual_events.cend());
        MPI_Send(own_data.data(), own_data.size(), mpiInfo.MPI_Type_StructuralPlasticityEvent, 0, 42, MPI_COMM_WORLD);
        structual_events.clear();
        own_data.resize(0);
        own_data.shrink_to_fit();
    }
}

template<typename P, typename L>
void MpiKestenSim<P, L>::mpiSaveResults()
{
    std::chrono::steady_clock::time_point t_now = std::chrono::steady_clock::now();
    auto t_since_begin = std::chrono::duration_cast<std::chrono::seconds>(t_now - t_begin).count();
    std::cout << steps << " steps " << " done in " << t_since_begin << " seconds" << std::endl;

    std::cout << "storing results..." << std::endl;
    std::ofstream output_file("./weights.txt");
    for (const auto& weight : w_all) {
        output_file << weight << "\n";
    }
    output_file.close();

    std::ofstream turnover_file("./turnover.txt");
    turnover_file << structual_events_all;
    turnover_file.close();
}

std::ostream& operator<<(std::ostream& ostream, const MpiInfo& info)
{
    ostream << "MpiInfo { " << info.rank << " of " << info.world_size
            << " with range [" << info.i_start << ", " << info.i_end << ")"
            << " }";
    return ostream;
}

template class MpiKestenSim<Parameters, KestenStep>;
