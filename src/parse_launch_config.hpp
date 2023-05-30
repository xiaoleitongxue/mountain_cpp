#ifndef PARSE_LAUNCH_CONFIG_H
#define PARSE_LAUNCH_CONFIG_H
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <partition_model.hpp>
using json = nlohmann::json;

typedef struct launch_parameter{
    std::string cfg;
    std::string weights;
    std::string filename;
    int stages;
    int workers;
    int frames;
    std::vector<partition_parameter> partition_params;
    server_address master_addr;
    std::vector<server_address> worker_addr;
}launch_parameter;
launch_parameter read_config(std::string filename);
#endif