#include "kestensimulation.h"
#include "main_shared.h"

int main(int argc, char** argv)
{
    Parameters p = params_from_arguments(argc, argv);

    KestenSimulation<Parameters, KestenStep> sim(p);
    while (sim.hasNextStep()) {
        sim.doStep();
    }
    sim.saveResults();

    return 0;
}
