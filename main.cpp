#include "kestensim.h"
#include "json.hpp"

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
    kestensim(p);

    return 0;
}
