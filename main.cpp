#include "json.hpp"
#include "kestensimulation.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " json-file" << std::endl;
        return 1;
    }

    std::string jsonfilepath = argv[1];
    std::ifstream jsonfile(jsonfilepath);
    nlohmann::json j;

    jsonfile >> j;
    Parameters p = j.get<Parameters>();

    KestenSimulation sim(p);
    while (sim.hasNextStep()) {
        sim.doStep();
    }
    sim.saveResults();

    return 0;
}
