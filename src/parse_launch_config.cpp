
#include <iostream>
#include "parse_launch_config.hpp"

launch_parameter read_init_config(const std::string& filename) {
  std::ifstream f(filename);
  json data = json::parse(f);
  launch_parameter launch_param;
  launch_param.cfg = data["cfg"];
  launch_param.weights = data["weights"];
  launch_param.filename = data["filename"];
  launch_param.workers = data["workers"];
  launch_param.stages = data["stages"];
  launch_param.frames = data["frames"];
  for (auto elem : data["partition_params"]) {
    partition_parameter partition_param{elem["partition_w"],
                                        elem["partition_h"], elem["partitions"],
                                        elem["from"], elem["to"]};
    launch_param.partition_params.push_back(partition_param);
  }
  for(auto elem : data["master_addr"]){
    server_address addr{elem["ip"], elem["port"]};
    launch_param.master_addr = addr;
  }

  for(auto elem : data["worker_addr"]){
    server_address addr{elem["ip"], elem["port"]};
    launch_param.worker_addr.push_back(addr);
  }
  return launch_param;

}
