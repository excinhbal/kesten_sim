#ifndef KESTEN_SIM_MAIN_SHARED_H
#define KESTEN_SIM_MAIN_SHARED_H

#include <iostream>
#include "json.hpp"
#include "kestensimulation.h"

Parameters params_from_arguments(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " json-file" << std::endl;
        exit(1);
    }

    std::string jsonfilepath = argv[1];
    std::ifstream jsonfile(jsonfilepath);
    nlohmann::json j;

    jsonfile >> j;
    return j.get<Parameters>();
}

QuadParameters quad_params_from_arguments(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " json-file" << std::endl;
        exit(1);
    }

    std::string jsonfilepath = argv[1];
    std::ifstream jsonfile(jsonfilepath);
    nlohmann::json j;

    jsonfile >> j;

    if (!j.contains("mu_alpha")) {
        std::cerr << "Bad parameter file without 'mu_alpha'." << std::endl;
        exit(1);
    }

    return j.get<QuadParameters>();
}

#endif //KESTEN_SIM_MAIN_SHARED_H
