#include <worker.hpp>
#include "inference_helper.hpp"
#include <chrono>
#include <iostream>
#include <parse_launch_config.hpp>
#include <string>
#include <thread>
#include <unistd.h>



using namespace std::chrono;
std::vector<std::pair<std::chrono::high_resolution_clock::time_point,
                      std::chrono::high_resolution_clock::time_point>>
    frame_time_point;
int main(int argc, char *argv[]) {
  std::string launch_json = argv[1];
  // prase json
  launch_parameter launch_param = read_config(launch_json);
  char *cfgfile = const_cast<char *>(launch_param.cfg.c_str());
  char *weights_file = const_cast<char *>(launch_param.weights.c_str());
  // load_network
  network net = parse_network_cfg_custom(cfgfile, 1, 1);

  for (int i = 0; i < launch_param.stages; ++i) {
    int from = launch_param.partition_params[i].from;
    int to = launch_param.partition_params[i].to;
    launch_param.partition_params[i].in_w = net.layers[from].w;
    launch_param.partition_params[i].in_h = net.layers[from].h;
    launch_param.partition_params[i].in_c = net.layers[from].c;
    launch_param.partition_params[i].out_w = net.layers[to].out_w;
    launch_param.partition_params[i].out_h = net.layers[to].out_h;
    launch_param.partition_params[i].out_c = net.layers[to].out_c;
  }
  std::vector<ftp_parameter> ftp_params =
      perform_ftp(launch_param.partition_params, launch_param.stages, net);
  frame_time_point =
      std::vector<std::pair<std::chrono::high_resolution_clock::time_point,
                            std::chrono::high_resolution_clock::time_point>>(
          launch_param.frames);
  std::string worker_type = argv[2];
  if (worker_type == "master") {
    network last_stage_net = parse_network_cfg_custom_whc(
        cfgfile, launch_param.partition_params[launch_param.stages - 1],
        ftp_params[launch_param.stages - 1], 0, 1, 1);
    load_weights_upto_subnet(
        &net, &last_stage_net, weights_file,
        launch_param.partition_params[launch_param.stages - 1].to,
        launch_param.partition_params[launch_param.stages - 1].from,
        launch_param.partition_params[launch_param.stages - 1].to);
    free_network(net);
    Master master{launch_param.master_addr.ip,
                  launch_param.master_addr.port,
                  launch_param.stages,
                  last_stage_net,
                  launch_param.frames,
                  launch_param.partition_params,
                  ftp_params,
                  launch_param.worker_addr};

    // start worker thread
    std::thread push_image_thread(&Master::m_push_image, &master);
    std::thread partition_image_thread(&Master::m_pritition_image, &master);
    std::thread inference_thread(&Master::m_inference, &master);
    std::thread send_data_packet_thread(&Master::m_send_data_packet, &master);
    std::thread recv_data_packet_thread(&Master::m_recv_data_packet, &master);
    std::thread merge_partition_thread(&Master::m_merge_partitions, &master);

    push_image_thread.join();
    partition_image_thread.join();
    merge_partition_thread.join();
    inference_thread.join();
    send_data_packet_thread.join();
    // recv_data_packet_thread.join();
    for (int i = 0; i < launch_param.frames; ++i) {
      auto diff = duration_cast<std::chrono::milliseconds>(
          frame_time_point[i].second - frame_time_point[i].first);
      std::cout << "frame Time " << i << " " << diff.count() << " milliseconds"
                << std::endl;
    }
  } else if (worker_type == "worker") {
    std::vector<std::vector<network>> sub_nets(launch_param.stages - 1);

    for (int i = 0; i < launch_param.stages - 1; ++i) {
      for (int j = 0; j < launch_param.partition_params[i].partitions; ++j) {
        network sub_net = parse_network_cfg_custom_whc(
            cfgfile, launch_param.partition_params[i], ftp_params[i], j, 1, 1);
        sub_nets[i].push_back(sub_net);
      }
    }

    // load weights
    load_sub_nets_weights(net, sub_nets, cfgfile, weights_file,
                          launch_param.stages - 1,
                          launch_param.partition_params);
    free_network(net);
    // flip weights
    flip_sub_nets_weights(sub_nets, launch_param.stages - 1,
                          launch_param.partition_params, ftp_params);

    // create worker object
    std::string worker_id = argv[3];
    int worker_id_ = std::stoi(worker_id);
    Worker worker{worker_id_, launch_param.worker_addr[worker_id_].ip,
                  launch_param.worker_addr[worker_id_].port, sub_nets,
                  launch_param.master_addr};
    // worker.m_receive_data_packet();
    // start worker thread
    std::thread recv_data_packet_thread(&Worker::m_recv_data_packet, &worker);
    std::thread send_data_packet_thread(&Worker::m_send_data_packet, &worker);
    std::thread inference_thread(&Worker::m_inference, &worker);

    inference_thread.join();
    recv_data_packet_thread.join();
    send_data_packet_thread.join();
  }
}
