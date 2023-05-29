#include "data_packet.hpp"
#include "partition_model.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/tensor.h>
#include <arpa/inet.h>
#include <c10/core/DeviceType.h>
#include <worker.hpp>

#include "stb_image.h"
#include <future>
#include <mutex>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static image load_image_stb(char *filename, int channels) {
  int w, h, c;
  unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
  if (!data)
    throw std::runtime_error("file not found");
  if (channels)
    c = channels;
  int i, j, k;
  image im = make_image(w, h, c);
  for (k = 0; k < c; ++k) {
    for (j = 0; j < h; ++j) {
      for (i = 0; i < w; ++i) {
        int dst_index = i + w * j + w * h * k;
        int src_index = k + c * i + c * w * j;
        im.data[dst_index] = (float)data[src_index] / 255.;
      }
    }
  }
  free(data);
  return im;
}

Worker::Worker(std::string ip, int port,
               std::vector<std::vector<network>> sub_nets) {
  m_ip = ip;
  m_port = port;
  m_sub_nets = sub_nets;
}

void Worker::m_inference() {
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU, 1)
                     .requires_grad(true);
  while (1) {
    std::unique_lock<std::mutex> lock1(m_prio_task_queue_mutex);
    Data_packet data_packet = m_prio_task_queue.top();
    m_prio_task_queue.pop();
    lock1.release();

    network net = m_sub_nets[data_packet.stage][data_packet.task_id];
    // convert tensor to array
    torch::Tensor cpu_tensor = data_packet.tensor.to(options);
    float *X = cpu_tensor.data_ptr<float>();
    float *out = network_predict(net, X);

    // convert array to torch::Tensor
    c10::IntArrayRef s = {net.layers[net.n - 1].c, net.layers[net.n - 1].h,
                          net.layers[net.n - 1].w};
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU, 1)
                       .requires_grad(true);
    torch::Tensor tensor = torch::from_blob(out, s, options);

    // create Data_packet
    Data_packet new_data_packet{data_packet.frame_seq,
                                data_packet.task_id,
                                data_packet.stage,
                                data_packet.from,
                                data_packet.to,
                                net.layers[net.n - 1].w,
                                net.layers[net.n - 1].h,
                                net.layers[net.n - 1].c,
                                net.layers[net.n - 1].w *
                                    net.layers[net.n - 1].h *
                                    net.layers[net.n - 1].c,
                                tensor};
    // push result to result queue
    std::unique_lock<std::mutex> lock2(m_prio_task_queue_mutex);
    m_prio_task_queue.push(new_data_packet);
  }
}

int Worker::m_receive_data_packet() {
  int server_fd, new_socket, valread;
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);
  char buffer[1024] = {0};

  // Creating socket file descriptor
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  // Forcefully attaching socket to the port 8080
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                 sizeof(opt))) {
    perror("setsockopt");
    exit(EXIT_FAILURE);
  }
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(m_port);

  // Forcefully attaching socket to the port 8080
  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }
  if (listen(server_fd, 3) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
  }
  if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                           (socklen_t *)&addrlen)) < 0) {
    perror("accept");
    exit(EXIT_FAILURE);
  }

  std::future<void> readFuture = std::async(std::launch::async, [new_socket]() {
    while (1) {
      int metadata_buffer[9];
      Data_packet data_packet;
      int valread = read(new_socket, metadata_buffer, sizeof(int) * 9);
      data_packet.frame_seq = metadata_buffer[0];
      data_packet.task_id = metadata_buffer[1];
      data_packet.stage = metadata_buffer[2];
      data_packet.from = metadata_buffer[3];
      data_packet.to = metadata_buffer[4];
      data_packet.w = metadata_buffer[5];
      data_packet.h = metadata_buffer[6];
      data_packet.c = metadata_buffer[7];
      data_packet.tensor_size = metadata_buffer[8];
      char tensor_buffer[data_packet.tensor_size];
      recv(new_socket, tensor_buffer, data_packet.tensor_size, MSG_WAITALL);
    }
  });

  std::future<void> sendFuture =
      std::async(std::launch::async, [this, new_socket]() {
        while (1) {
          std::unique_lock<std::mutex> lock(m_prio_result_queue_mutex);
          Data_packet data_packet = m_prio_task_queue.top();
          m_prio_task_queue.pop();
          lock.release();
          void *serialized_data_packet = serialize_data_packet(data_packet);
          send(new_socket, serialized_data_packet, data_packet.tensor_size, 0);
          printf("Hello message sent\n");
        }
      });

  // closing the connected socket
  close(new_socket);
  // closing the listening socket
  shutdown(server_fd, SHUT_RDWR);
  return 0;
}

Master::Master(std::string ip, int port, network net, network last_stage_net,
               int frames, std::vector<partition_parameter> partition_params,
               std::vector<ftp_parameter> ftp_params,
               std::vector<server_address> server_addresses)

{
  m_ip = ip;
  m_port = port;
  m_net = net;
  m_last_stage_net = last_stage_net;
  m_frames = frames;
  m_partition_params = partition_params;
  m_ftp_params = ftp_params;
  m_server_addresses = server_addresses;
}

LIB_API image_t Master::load_image(std::string image_filename) {
  char *input = const_cast<char *>(image_filename.c_str());
  image im = load_image_stb(input, 3);

  image_t img;
  img.c = im.c;
  img.data = im.data;
  img.h = im.h;
  img.w = im.w;

  return img;
}

void Master::m_push_image() {
  std::string image_path;
  for (int i = 0; i < m_frames; ++i) {
    auto img = load_image(image_path);
    image im;
    im.c = img.c;
    im.data = img.data;
    im.h = img.h;
    im.w = img.w;
    image sized;
    if (m_net.w == im.w && m_net.h == im.h) {
      sized = make_image(im.w, im.h, im.c);
      memcpy(sized.data, im.data, im.w * im.h * im.c * sizeof(float));
    } else
      sized = resize_image(im, m_net.w, m_net.h);
    // covert image to torch tensor
    c10::IntArrayRef s = {sized.c, sized.h, sized.w};
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU, 1)
                       .requires_grad(true);
    torch::Tensor tensor = torch::from_blob(sized.data, s, options);

    Data_packet data_packet{1,       0,       0,       0, 0,
                            sized.w, sized.h, sized.c, 0, tensor};

    // push image to queue
    std::unique_lock<std::mutex> lock(m_prio_image_queue_mutex);
    m_prio_image_queue.push(data_packet);
  }
}

void Master::m_pritition_image() {
  while (1) {
    std::unique_lock<std::mutex> lock(m_prio_image_queue_mutex);
    Data_packet data_packet = m_prio_image_queue.top();
    m_prio_image_queue.pop();
    lock.release();

    int frame_seq = data_packet.frame_seq;
    int stage = data_packet.stage;
    // crop image
    for (int i = 0; i < m_ftp_params[stage].partitions_w; ++i) {
      for (int j = 0; j < m_ftp_params[stage].partitions_h; ++j) {
        int task_id = m_ftp_params[stage].task_ids[i][j];
        int dw1 = m_ftp_params[0].input_tiles[task_id][0].w1;
        int dw2 = m_ftp_params[0].input_tiles[task_id][0].w2;
        int dh1 = m_ftp_params[0].input_tiles[task_id][0].h1;
        int dh2 = m_ftp_params[0].input_tiles[task_id][0].h2;
        int c = m_ftp_params[0].input_tiles[task_id][0].c;
        // crop
        torch::Tensor partition = data_packet.tensor.index(
            {at::indexing::Slice(0, c), at::indexing::Slice(dh1, dh2 + 1),
             at::indexing::Slice(dw1, dw2 + 1)});
        // flip
        switch (task_id) {
        case 0:
          break;
        case 1:
          // flip lr
          partition.flip(2);
          break;
        case 2:
          // flip ud
          partition.flip(1);
          break;
        case 3:
          partition.flip({1, 2});
          break;
        }

        Data_packet data_packet{frame_seq,
                                task_id,
                                stage,
                                m_partition_params[0].from,
                                m_partition_params[0].to,
                                dw2 - dw1 + 1,
                                dh2 - dh1 + 1,
                                c,
                                0,
                                partition};
        std::unique_lock<std::mutex> lock(m_prio_task_queue_mutex);
        m_prio_task_queue.push(data_packet);
      }
    }
  }
}

void Master::m_merge_partitions() {
  while (1) {

    // collect all partitions
    // processs frame by frame

    // which layer needs to be merged
    int counts = 0;
    std::unique_lock<std::mutex> lock(m_prio_partition_inference_result_mutex);
    Data_packet data_packet = m_prio_partition_inference_result_queue.top();
    lock.release();

    int frame_seq = data_packet.frame_seq;
    int task_id = data_packet.task_id;
    int stage = data_packet.stage;
    int from = data_packet.from;
    int to = data_packet.to;

    // create merged image
    c10::IntArrayRef s = {m_net.layers[to].c, m_net.layers[to].h,
                          m_net.layers[to].w};
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU, 1)
                       .requires_grad(true);
    torch::Tensor merged = torch::rand(s, options);

    while (counts <= m_partition_params[stage].partitions) {
      std::unique_lock<std::mutex> lock(
          m_prio_partition_inference_result_mutex);
      Data_packet data_packet = m_prio_partition_inference_result_queue.top();
      if (data_packet.frame_seq != frame_seq || data_packet.stage != stage) {
        continue;
      }
      m_prio_partition_inference_result_queue.pop();
      lock.release();

      int frame_seq = data_packet.frame_seq;
      int task_id = data_packet.task_id;
      int stage = data_packet.stage;
      int from = data_packet.from;
      int to = data_packet.to;
      torch::Tensor partiton = data_packet.tensor;
      int fused_layers = m_ftp_params[stage].fused_layers;
      int dw1 = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].w1;
      int dw2 = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].w2;
      int dh1 = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].h1;
      int dh2 = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].h2;
      int w = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].w;
      int h = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].h;
      int c = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].c;
      // crop padding
      torch::Tensor cropped_partition =
          partiton.index({at::indexing::Slice(0, c), at::indexing::Slice(0, h),
                          at::indexing::Slice(0, w)});
      // flip data

      switch (task_id) {
      case 0:
        break;
      case 1:
        // flip lr
        cropped_partition.flip(2);
        break;
      case 2:
        // flip ud
        cropped_partition.flip(1);
        break;
      case 3:
        cropped_partition.flip({1, 2});
        break;
      }
      // stitch partition to original image
      merged.index({at::indexing::Slice(0, c),
                    {at::indexing::Slice(dh1, dh2 + 1)},
                    {at::indexing::Slice(dw1, dw2 + 1)}}) = cropped_partition;
      Data_packet new_data_packet{frame_seq,
                                  task_id,
                                  stage + 1,
                                  m_partition_params[stage].from,
                                  m_partition_params[stage].to,
                                  dw2 - dw1 + 1,
                                  dh2 - dh1 + 1,
                                  c,
                                  0,
                                  cropped_partition};
      if (stage < m_stages - 1) {
        std::unique_lock<std::mutex> lock1(m_prio_image_queue_mutex);
        m_prio_image_queue.push(new_data_packet);
      } else {
        std::unique_lock<std::mutex> lock1(m_prio_merged_result_mutex);
        m_prio_merged_result_queue.push(new_data_packet);
      }
    }
  }
}

int Master::m_send_data_packet() {
  // connect to all worker
  std::vector<int> client_fds;
  auto connected_to_all_worker = [=](std::vector<int> &client_fds) {
    for (auto worker_address : m_server_addresses) {
      int status, valread, client_fd;
      struct sockaddr_in serv_addr;
      if ((client_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
      }
      serv_addr.sin_family = AF_INET;
      serv_addr.sin_port = htons(worker_address.port);

      // Convert IPv4 and IPv6 addresses from text to binary
      // form
      if (inet_pton(AF_INET, const_cast<char *>(worker_address.ip.c_str()),
                    &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
      }

      if ((status = connect(client_fd, (struct sockaddr *)&serv_addr,
                            sizeof(serv_addr))) < 0) {
        printf("\nConnection Failed \n");
        return -1;
      }
      client_fds.push_back(client_fd);
    }
    return 0;
  };

  std::future<void> sendFuture =
      std::async(std::launch::async, [this, client_fds]() {
        int counts = 0;
        while (1) {
          int send_to = counts % client_fds.size();
          std::unique_lock<std::mutex> lock(m_prio_task_queue_mutex);
          Data_packet data_packet = m_prio_task_queue.top();
          m_prio_task_queue.pop();
          lock.release();
          void *serialized_data_packet = serialize_data_packet(data_packet);
          send(client_fds[send_to], serialized_data_packet,
               data_packet.tensor_size, 0);
          printf("Hello message sent\n");
          counts++;
        }
      });

  // handle reply
  std::future<void> readFuture =
      std::async(std::launch::async, [this, client_fds]() {
        int counts = 0;
        while (1) {
          int receive_from = counts % client_fds.size();
          int metadata_buffer[9];
          Data_packet data_packet;
          int valread =
              read(client_fds[receive_from], metadata_buffer, sizeof(int) * 9);
          data_packet.frame_seq = metadata_buffer[0];
          data_packet.task_id = metadata_buffer[1];
          data_packet.stage = metadata_buffer[2];
          data_packet.from = metadata_buffer[3];
          data_packet.to = metadata_buffer[4];
          data_packet.w = metadata_buffer[5];
          data_packet.h = metadata_buffer[6];
          data_packet.c = metadata_buffer[7];
          data_packet.tensor_size = metadata_buffer[8];
          char tensor_buffer[data_packet.tensor_size];
          recv(client_fds[receive_from], tensor_buffer, data_packet.tensor_size,
               MSG_WAITALL);
          c10::IntArrayRef s = {data_packet.c, data_packet.h, data_packet.w};
          auto options = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .layout(torch::kStrided)
                             .device(torch::kCPU, 1)
                             .requires_grad(true);
          torch::Tensor tensor = torch::from_blob(tensor_buffer, s, options);
          // push to queue
          m_prio_partition_inference_result_queue.push(data_packet);
        }
      });

  // closing the connected socket
  // close(client_fds[rec]);
  return 0;
}

inline static void *serialize_data_packet(Data_packet &data_packet) {
  std::ostringstream stream;
  // torch::save(data_packet.tensor, stream);
  const std::string str = stream.str();
  const int length = str.length();
  data_packet.tensor_size = length;

  char *buffer = new char(sizeof(int) * 9 + length);

  std::memcpy(buffer, &data_packet, sizeof(int) * 9);
  char *p = buffer + sizeof(int) * 9;
  std::memcpy(p, str.data(), length);
  return buffer;
}
// inference final stage
void Master::m_inference() {
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCPU, 1)
                     .requires_grad(true);
  while (1) {
    std::unique_lock<std::mutex> lock(m_prio_merged_result_mutex);
    Data_packet data_packet = m_prio_merged_result_queue.top();
    m_prio_merged_result_queue.pop();
    lock.release();
    // convert tensor to array
    torch::Tensor cpu_tensor = data_packet.tensor.to(options);
    float *X = cpu_tensor.data_ptr<float>();
    float *out = network_predict(m_last_stage_net, X);

    // convert array to torch::Tensor
    c10::IntArrayRef s = {m_last_stage_net.layers[m_last_stage_net.n - 1].c,
                          m_last_stage_net.layers[m_last_stage_net.n - 1].h,
                          m_last_stage_net.layers[m_last_stage_net.n - 1].w};
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU, 1)
                       .requires_grad(true);
    torch::Tensor tensor = torch::from_blob(out, s, options);

    // create Data_packet
    Data_packet new_data_packet{
        data_packet.frame_seq,
        data_packet.task_id,
        data_packet.stage,
        data_packet.from,
        data_packet.to,
        m_last_stage_net.layers[m_last_stage_net.n - 1].w,
        m_last_stage_net.layers[m_last_stage_net.n - 1].h,
        m_last_stage_net.layers[m_last_stage_net.n - 1].c,
        m_last_stage_net.layers[m_last_stage_net.n - 1].w *
            m_last_stage_net.layers[m_last_stage_net.n - 1].h *
            m_last_stage_net.layers[m_last_stage_net.n - 1].c,
        tensor};
    std::cout << out << std::endl;
  }
}