#ifndef WORKER_HPP
#define WORKER_HPP
#include "partition_model.hpp"
#include <boost/archive/binary_oarchive.hpp>
#include <darknet.h>
#include <data_packet.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <vector>

typedef struct server_address {
  std::string ip;
  int port;
} erver_address;

class Worker {
private:
  std::string m_ip;
  int m_port;
  std::vector<std::vector<network>> m_sub_nets;
  std::priority_queue<Data_packet> m_prio_task_queue;
  std::priority_queue<Data_packet> m_prio_result_queue;

public:
  Worker(std::string ip, int port, std::vector<std::vector<network>> sub_nets);
  void m_inference();
  void m_sent_data_packet();
  void m_receive_data_packet();
};

class Master {
private:
  std::string m_ip;
  int m_port;
  network m_net;
  int m_frames;

  // mutex for thread synchronization
  std::mutex m_prio_task_queue_mutex;
  std::mutex m_prio_image_queue_mutex;
  std::mutex m_prio_partition_inference_result_mutex;
  std::mutex m_prio_merged_result_mutex;

  std::priority_queue<Data_packet> m_prio_task_queue;
  std::priority_queue<Data_packet> m_prio_image_queue;
  std::priority_queue<Data_packet> m_prio_partition_inference_result_queue;
  std::priority_queue<Data_packet> m_prio_merged_result_queue;

  std::vector<partition_parameter> m_partition_params;
  std::vector<ftp_parameter> m_ftp_params;
  std::vector<server_address> m_server_addresses;

public:
  Master(std::string ip, int port, network net, int frames,
         std::vector<partition_parameter> partition_params,
         std::vector<ftp_parameter> ftp_params,
         std::vector<server_address> server_addresses);
  void m_pritition_image();
  void m_merge_partitions();
  void m_inference();
  int m_send_data_packet();
  void m_receive_data_packet();
  void m_push_image();
  void *serialize_data_packet(Data_packet &data_packet);
  static LIB_API image_t load_image(std::string image_filename);
};

#endif