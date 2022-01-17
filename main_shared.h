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

#endif //KESTEN_SIM_MAIN_SHARED_H
