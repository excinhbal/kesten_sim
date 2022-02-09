#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "kestensimulation.h"

std::string readfile(std::string path) {
    std::ifstream t(path);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

TEST_CASE("End-to-end seed 193945") {
    Parameters p;
    KestenSimulation<Parameters, KestenStep> sim(p);
    while (sim.hasNextStep()) {
        sim.doStep();
    }
    sim.saveResults();

    auto weights = readfile("./weights.txt");
    auto weights_actual = readfile("../testdata/weights.txt");
    REQUIRE(weights == weights_actual);

    auto turnover = readfile("./turnover.txt");
    auto turnover_actual = readfile("../testdata/turnover.txt");
    REQUIRE(turnover == turnover_actual);
}
