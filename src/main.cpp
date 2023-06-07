#include <worker.hpp>
#include "inference_helper.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <parse_launch_config.hpp>
#include <string>
#include <thread>
#include <unistd.h>
#include <chrono>
using namespace std;
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
  // network *net = load_network(cfgfile, weights_file, 0);
  network net = parse_network_cfg_custom(cfgfile, 1, 1);
  if (weights_file) {
    load_weights(&net, weights_file);
  }
  // ftp
  std::vector<ftp_parameter> ftp_params =
      perform_ftp(launch_param.partition_params, launch_param.stages, net);
  // generate sub_net
  std::vector<std::vector<network>> sub_nets = generate_sub_network(
      cfgfile, ftp_params, launch_param.stages, launch_param.partition_params);
  // load weights
  load_sub_nets_weights(sub_nets, cfgfile, weights_file, launch_param.stages,
                        launch_param.partition_params);
  // flip weights
  flip_sub_nets_weights(sub_nets, launch_param.stages,
                        launch_param.partition_params, ftp_params);
  frame_time_point = std::vector<std::pair<std::chrono::high_resolution_clock::time_point,
                      std::chrono::high_resolution_clock::time_point>>(100);
  std::string worker_type = argv[2];
  if (worker_type == "master") {
    Master master = Master{launch_param.master_addr.ip,
                           launch_param.master_addr.port,
                           net,
                           launch_param.stages,
                           sub_nets[launch_param.stages - 1][0],
                           launch_param.frames,
                           launch_param.partition_params,
                           ftp_params,
                           launch_param.worker_addr,
                           sub_nets};

    // master.m_push_image(launch_param.filename);
    // master.m_pritition_image();
    // // std::thread send_data_packet_thread(&Master::m_send_data_packet,
    // &master); sleep(5); master.m_merge_partitions();

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
    

    for (int i = 0; i < launch_param.frames; ++i) {
      auto diff =
          duration_cast<std::chrono::milliseconds> (frame_time_point[i].second -
          frame_time_point[i].first);
      std::cout << "frame Time " << i << " " << diff.count() << " milliseconds" << std::endl;
    }
    recv_data_packet_thread.join();

    // inform worker to termial

  } else if (worker_type == "worker") {
    // create worker object
    std::string worker_id = argv[3];
    int worker_id_ = std::stoi(worker_id);
    Worker worker = Worker{launch_param.worker_addr[worker_id_].ip,
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