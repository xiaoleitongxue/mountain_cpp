#include "inference_helper.hpp"
#include "partition_model.hpp"
#include <darknet.h>
#include <iostream>
#include <string>
#include <parse_launch_config.hpp>
int main(int argc, char *argv[]){
    std::string  launch_json = argv[1];
    // prase json
    launch_parameter launch_param = read_config(launch_json);
    char *cfgfile = const_cast<char *>(launch_param.cfg.c_str());
    char *weights_file = const_cast<char *>(launch_param.weights.c_str());
    //load_network
    network *net = load_network(cfgfile, weights_file, 0);
    //ftp
    std::vector<ftp_parameter> ftp_params = perform_ftp(launch_param.partition_params, launch_param.stages, *net);
    //generate sub_net
    std::vector<std::vector<network>> sub_nets = generate_sub_network(cfgfile, ftp_params, launch_param.stages, launch_param.partition_params);
    //load weights
    load_sub_nets_weights(sub_nets, cfgfile, weights_file, launch_param.stages, launch_param.partition_params);
    //flip weights
    flip_sub_nets_weghts(sub_nets, launch_param.stages, launch_param.partition_params, ftp_params);

    
}