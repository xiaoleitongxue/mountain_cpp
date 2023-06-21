#ifndef WORKER_HPP_LLL
#define WORKER_HPP_LLL

#include <ATen/core/TensorBody.h>
#include <arpa/inet.h>
#include <c10/core/DeviceType.h>


#include <data_packet.hpp>
#include <torch/serialize/input-archive.h>
#include <torch/torch.h>

#include "partition_model.hpp"
#include <darknet.h>
#include "yolo_v2_class.hpp"
#include <mutex>

extern std::vector<std::pair<std::chrono::high_resolution_clock::time_point,
                             std::chrono::high_resolution_clock::time_point>>
    frame_time_point;
class Compare {
public:
  bool operator()(const Data_packet &a, const Data_packet &b) {
    if (a.frame_seq == b.frame_seq) {
      return a.stage >= b.stage;
    }
    return a.frame_seq > b.frame_seq;
  }
};

class Worker {
private:
  int m_worker_id;
  std::string m_ip;
  int m_port;
  std::mutex m_prio_task_queue_mutex;
  std::mutex m_prio_result_queue_mutex;
  struct server_address m_master_addr;
  std::vector<std::vector<network>> m_sub_nets;
  std::priority_queue<Data_packet, std::vector<Data_packet>, Compare>
      m_prio_task_queue;
  std::priority_queue<Data_packet, std::vector<Data_packet>, Compare>
      m_prio_result_queue;

  std::thread m_inference_thread;
  std::thread m_receive_data_packet_thread;

public:
  Worker(int worker_id, std::string ip, int port,
         std::vector<std::vector<network>> sub_nets,
         struct server_address master_addr);
  Worker() = delete;
  Worker(const Worker &) = delete;
  Worker(const Worker &&) = delete;
  void m_inference();
  int m_recv_data_packet();
  int m_send_data_packet();
};

class Master {
private:
  std::string m_ip;
  int m_port;
  network m_last_stage_net;
  int m_frames;
  int m_stages;
  int exit_flag;
  std::vector<torch::Tensor> correct_tensor;
  // mutex for thread synchronization
  std::mutex m_prio_task_queue_mutex;
  std::mutex m_prio_image_queue_mutex;
  std::mutex m_prio_partition_inference_result_mutex;
  std::mutex m_prio_merged_result_mutex;

  std::priority_queue<Data_packet, std::vector<Data_packet>, Compare>
      m_prio_task_queue;
  std::priority_queue<Data_packet, std::vector<Data_packet>, Compare>
      m_prio_image_queue;
  std::queue<Data_packet> m_prio_partition_inference_result_queue;
  std::priority_queue<Data_packet, std::vector<Data_packet>, Compare>
      m_prio_merged_result_queue;

  std::vector<partition_parameter> m_partition_params;
  std::vector<ftp_parameter> m_ftp_params;
  std::vector<server_address> m_server_addresses;
  // std::vector<std::vector<network>> m_sub_nets;

  std::thread m_pritition_image_thread;
  std::thread m_merge_partitions_thread;
  std::thread m_inference_thread;
  std::thread m_sned_data_packet_thread;
  std::thread m_push_image_thread;

public:
  Master(std::string ip, int port, int stages, network last_stage_net,
         int frames, std::vector<partition_parameter> partition_params,
         std::vector<ftp_parameter> ftp_params,
         std::vector<server_address> server_addresses);
  // default constructor
  Master() = delete;
  Master(const Master &) = delete;
  Master(const Master &&) = delete;
  void m_pritition_image();
  void m_merge_partitions();
  void m_inference();
  int m_send_data_packet(int client_fd, int num);
  int m_recv_data_packet();
  void m_push_image(int frame_seq);
  static LIB_API image_t load_image(std::string image_filename);
};

inline static void *serialize_data_packet(Data_packet &data_packet);

#endif