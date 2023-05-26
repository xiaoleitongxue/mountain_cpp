#include "data_packet.hpp"
#include "partition_model.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/tensor.h>
#include <arpa/inet.h>
#include <darknet.h>
#include <future>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stb_image.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <worker.hpp>
#include <yolo_v2_class.hpp>
#define PORT 8080
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
  Data_packet data_packet = m_prio_task_queue.top();
  network net = m_sub_nets[data_packet.stage][data_packet.task_id];
  // float *out = network_predict(net, data_packet.data);
}

void Worker::m_sent_data_packet() {}

void Worker::m_receive_data_packet() {}

Master::Master(std::string ip, int port, network net, int frames,
               std::vector<partition_parameter> partition_params,
               std::vector<ftp_parameter> ftp_params,
               std::vector<server_address> server_addresses) {
                m_ip = ip;
                m_port = port;
                m_net = net;
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
  for (int i = 0; i < frames; ++i) {
    auto img = load_image(image_path);
    image im;
    im.c = img.c;
    im.data = img.data;
    im.h = img.h;
    im.w = img.w;
    image sized;
    if (net.w == im.w && net.h == im.h) {
      sized = make_image(im.w, im.h, im.c);
      memcpy(sized.data, im.data, im.w * im.h * im.c * sizeof(float));
    } else
      sized = resize_image(im, net.w, net.h);
    // covert image to torch tensor
    c10::IntArrayRef s = {sized.c, sized.h, sized.w};
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCUDA, 1)
                       .requires_grad(true);
    torch::Tensor tensor = torch::from_blob(sized.data, s, options);

    Data_packet data_packet{1, 0, 0, 0, 0, sized.w, sized.h, sized.c, tensor};

    // push image to queue
    std::unique_lock<std::mutex> lock(m_prio_image_queue_mutex);
    m_prio_image_queue.push(data_packet);
  }
}

void Master::m_pritition_image() {
  std::unique_lock<std::mutex> lock(m_prio_image_queue_mutex);

  Data_packet data_packet = m_prio_image_queue.top();
  m_prio_image_queue.pop();
  lock.release();

  int frame_seq = data_packet.frame_seq;
  // int task_id = data_packet.task_id;
  int stage = data_packet.stage;
  int from = data_packet.from;
  int to = data_packet.to;
  // crop image
  for (int i = 0; i < m_ftp_params[stage].partitions_w; ++i) {
    for (int j = 0; j < m_ftp_params[stage].partitions_h; ++j) {
      int task_id = m_ftp_params[stage].task_ids[i][j];
      int dw1 = m_ftp_params[stage].input_tiles[task_id][from].w1;
      int dw2 = m_ftp_params[stage].input_tiles[task_id][from].w2;
      int dh1 = m_ftp_params[stage].input_tiles[task_id][from].h1;
      int dh2 = m_ftp_params[stage].input_tiles[task_id][from].h2;
      int c = m_ftp_params[stage].input_tiles[task_id][from].c;
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
                              m_partition_params[stage].from,
                              m_partition_params[stage].to,
                              dw2 - dw1 + 1,
                              dh2 - dh1 + 1,
                              c,
                              partition};
      std::unique_lock<std::mutex> lock(m_prio_task_queue_mutex);
      m_prio_task_queue.push(data_packet);
    }
  }
}

void Master::m_merge_partitions() {
  while (1) {
    int counts = 0;
    std::unique_lock<std::mutex> lock(m_prio_partition_inference_result_mutex);
    Data_packet data_packet = m_prio_partition_inference_result_queue.top();
    m_prio_partition_inference_result_queue.pop();
    lock.release();

    int frame_seq = data_packet.frame_seq;
    int task_id = data_packet.task_id;
    int stage = data_packet.stage;
    int from = data_packet.from;
    int to = data_packet.to;

    // create merged image
    c10::IntArrayRef s = {net.layers[to].c, net.layers[to].h, net.layers[to].w};
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCUDA, 1)
                       .requires_grad(true);
    torch::Tensor merged = torch::rand(s, options);

    while (counts <= 4) {
      Data_packet data_packet = m_prio_partition_inference_result_queue.top();

      int frame_seq = data_packet.frame_seq;
      int task_id = data_packet.task_id;
      int stage = data_packet.stage;
      int from = data_packet.from;
      int to = data_packet.to;
      torch::Tensor partiton = data_packet.tensor;

      int dw1 = m_ftp_params[stage].output_tiles[task_id][to].w1;
      int dw2 = m_ftp_params[stage].output_tiles[task_id][to].w2;
      int dh1 = m_ftp_params[stage].output_tiles[task_id][to].h1;
      int dh2 = m_ftp_params[stage].output_tiles[task_id][to].h2;
      int w = m_ftp_params[stage].output_tiles[task_id][to].w;
      int h = m_ftp_params[stage].output_tiles[task_id][to].h;
      int c = m_ftp_params[stage].output_tiles[task_id][to].c;
      // crop padding
      torch::Tensor cropped_partition = partiton.index(
          {at::indexing::Slice(0, c), at::indexing::Slice(0, h + 1),
           at::indexing::Slice(0, w + 1)});
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
                                  stage,
                                  m_partition_params[stage].from,
                                  m_partition_params[stage].to,
                                  dw2 - dw1 + 1,
                                  dh2 - dh1 + 1,
                                  c,
                                  cropped_partition};
      std::unique_lock<std::mutex> lock(m_prio_image_queue_mutex);
      m_prio_image_queue.push(new_data_packet);
    }
  }
}

int Master::m_send_data_packet() {
  int status, valread, client_fd;
  struct sockaddr_in serv_addr;
  char *hello = "Hello from client";
  char buffer[1024] = {0};
  if ((client_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    printf("\n Socket creation error \n");
    return -1;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(PORT);

  // Convert IPv4 and IPv6 addresses from text to binary
  // form
  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
    printf("\nInvalid address/ Address not supported \n");
    return -1;
  }

  if ((status = connect(client_fd, (struct sockaddr *)&serv_addr,
                        sizeof(serv_addr))) < 0) {
    printf("\nConnection Failed \n");
    return -1;
  }

  std::future<void> sendFuture = std::async(std::launch::async, [=]() {
    while (1) {
      std::unique_lock<std::mutex> lock(m_prio_task_queue_mutex);
      Data_packet data_packet = m_prio_task_queue.top();
      m_prio_task_queue.pop();
      lock.release();
      void *serialized_data_packet = serialize_data_packet(data_packet);
      send(client_fd, serialized_data_packet, data_packet.tensor_size, 0);
      printf("Hello message sent\n");
    }
  });

  std::future<void> readFuture = std::async(std::launch::async, [client_fd]() {
    while (1) {
      int metadata_buffer[9];
      Data_packet data_packet;
      int valread = read(client_fd, metadata_buffer, sizeof(int) * 9);
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
      recv(client_fd, tensor_buffer, data_packet.tensor_size, MSG_WAITALL);
    }
  });

  // closing the connected socket
  close(client_fd);
  return 0;
}

void *Master::serialize_data_packet(Data_packet &data_packet) {
  std::ostringstream stream;
  torch::save(data_packet.tensor, stream);
  const std::string str = stream.str();
  const int length = str.length();
  data_packet.tensor_size = length;
  // const char *buffer = str.c_str();
  char *buffer = new char(sizeof(int) * 9 + length);
  ;
  std::memcpy(buffer, &data_packet, sizeof(int) * 9);
  char *p = buffer + sizeof(int) * 9;
  std::memcpy(p, str.data(), length);
  return buffer;
}