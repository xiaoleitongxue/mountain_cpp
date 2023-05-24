#include "partition_model.hpp"
#include <iostream>
#include <parse_launch_config.hpp>
launch_parameter read_config(std::string filename) {
  std::ifstream f(filename);
  json data = json::parse(f);
  launch_parameter launch_param;
  launch_param.cfg = data["cfg"];
  launch_param.weights = data["weights"];
  launch_param.filename = data["filename"];
  launch_param.workers = data["workers"];
  launch_param.stages = data["stages"];
  for (auto elem : data["partition_params"]) {
    partition_parameter partition_param{elem["partition_w"],
                                        elem["partition_h"], elem["partitions"],
                                        elem["from"], elem["to"]};
    launch_param.partition_params.push_back(partition_param);
  }
  for(auto elem : data["master_addr"]){
    server_addr addr{elem["ip"], elem["port"]};
  }

  for(auto elem : data["worker_addr"]){
    server_addr addr{elem["ip"], elem["port"]};
  }
  return launch_param;

}
