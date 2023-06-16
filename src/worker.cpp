#include "data_packet.hpp"
#include <c10/core/DeviceType.h>
#include <chrono>
#include <iostream>
#include <worker.hpp>

#include <cassert>
#include <darknet.h>
#include <string>

#include "partition_model.hpp"
#include "stb_image.h"

#include <errno.h>
#include <future>
#include <mutex>
extern "C"
{
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
}

int make_socket(uint16_t port)
{
  int sock;
  struct sockaddr_in name;

  /* Create the socket. */
  sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0)
  {
    perror("socket");
    exit(EXIT_FAILURE);
  }

  /* Give the socket a name. */
  name.sin_family = AF_INET;
  name.sin_port = htons(port);
  name.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(sock, (struct sockaddr *)&name, sizeof(name)) < 0)
  {
    perror("bind");
    exit(EXIT_FAILURE);
  }

  return sock;
}

void init_sockaddr(struct sockaddr_in *name, const char *hostname, int port)
{
  struct hostent *hostinfo;

  name->sin_family = AF_INET;
  name->sin_port = htons(port);
  hostinfo = gethostbyname(hostname);
  if (hostinfo == NULL)
  {
    fprintf(stderr, "Unknown host %s.\n", hostname);
    exit(EXIT_FAILURE);
  }
  name->sin_addr = *(struct in_addr *)hostinfo->h_addr;
}

void show_console_result(std::vector<bbox_t> const result_vec,
                         std::vector<std::string> const obj_names,
                         int frame_id = -1)
{
  if (frame_id >= 0)
    std::cout << " Frame: " << frame_id << std::endl;
  for (auto &i : result_vec)
  {
    if (obj_names.size() > i.obj_id)
      std::cout << obj_names[i.obj_id] << " - ";
    std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
              << ", w = " << i.w << ", h = " << i.h << std::setprecision(3)
              << ", prob = " << i.prob << std::endl;
  }
}

std::vector<std::string> objects_names_from_file(std::string const filename)
{
  std::ifstream file(filename);
  std::vector<std::string> file_lines;
  if (!file.is_open())
    return file_lines;
  for (std::string line; getline(file, line);)
    file_lines.push_back(line);
  std::cout << "object names loaded \n";
  return file_lines;
}

static image load_image_stb(char *filename, int channels)
{
  int w, h, c;
  unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
  if (!data)
    throw std::runtime_error("file not found");
  if (channels)
    c = channels;
  int i, j, k;
  image im = make_image(w, h, c);
  for (k = 0; k < c; ++k)
  {
    for (j = 0; j < h; ++j)
    {
      for (i = 0; i < w; ++i)
      {
        int dst_index = i + w * j + w * h * k;
        int src_index = k + c * i + c * w * j;
        im.data[dst_index] = (float)data[src_index] / 255.;
      }
    }
  }
  free(data);
  return im;
}

Worker::Worker(int worker_id, std::string ip, int port,
               std::vector<std::vector<network>> sub_nets,
               struct server_address master_addr)
{
  m_worker_id = worker_id;
  m_ip = ip;
  m_port = port;
  m_sub_nets = sub_nets;
  m_master_addr = master_addr;
}

void Worker::m_inference()
{
  while (1)
  {

    std::unique_lock<std::mutex> lock1(m_prio_task_queue_mutex);
    if (m_prio_task_queue.empty())
    {
      continue;
    }
    Data_packet data_packet = m_prio_task_queue.top();
    m_prio_task_queue.pop();
    lock1.unlock();
    
    // auto start = std::chrono::high_resolution_clock::now();
    network net = m_sub_nets[data_packet.stage][data_packet.task_id];
    // assert(data_packet.tensor.sizes()[0] == net.c);
    // assert(data_packet.tensor.sizes()[1] == net.h);
    // assert(data_packet.tensor.sizes()[2] == net.w);
    // convert tensor to array
    float *X = data_packet.tensor.data_ptr<float>();
    // std::cout << "start inference !" << std::endl;
    float *out = network_predict(net, X);
    // std::cout << "output: " << *out << std::endl;
    // convert array to torch::Tensor
    c10::IntArrayRef s = {net.layers[net.n - 1].out_c,
                          net.layers[net.n - 1].out_h,
                          net.layers[net.n - 1].out_w};

    torch::Tensor tensor = torch::from_blob(out, s).clone();
    // std::cout << net.layers[net.n - 1].out_c << " "
    //           << net.layers[net.n - 1].out_h << " "
    //           << net.layers[net.n - 1].out_w << std::endl;
    // assert(tensor.sizes()[0] == net.layers[net.n - 1].out_c);
    // assert(tensor.sizes()[1] == net.layers[net.n - 1].out_h);
    // assert(tensor.sizes()[2] == net.layers[net.n - 1].out_w);

    // std::ostringstream stream;
    // torch::save(tensor, stream);
    // const std::string str = stream.str();
    // const int length = str.length();

    // create Data_packet
    Data_packet new_data_packet{data_packet.frame_seq,
                                data_packet.task_id,
                                data_packet.stage,
                                data_packet.from,
                                data_packet.to,
                                net.layers[net.n - 1].out_w,
                                net.layers[net.n - 1].out_h,
                                net.layers[net.n - 1].out_c,
                                0,
                                tensor};
    // push result to result queue
    std::unique_lock<std::mutex> lock2(m_prio_result_queue_mutex);
    m_prio_result_queue.push(new_data_packet);
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // std::cout << "Time taken by function: " << duration.count()
    //           << " milliseconds" << std::endl;
    // std::cout << "end inference !" << std::endl;
  }
}
int Worker::m_send_data_packet()
{
  int sock;
  struct sockaddr_in servername;
  /* Create the socket. */
  sock = socket(AF_INET, SOCK_STREAM, 0);
  init_sockaddr(&servername, m_master_addr.ip.c_str(), m_master_addr.port);
  if (sock < 0)
  {
    perror("socket (client)");
    exit(EXIT_FAILURE);
  }
  /* Connect to the server. */
  while (1)
  {
    if (0 > connect(sock, (struct sockaddr *)&servername, sizeof(servername)))
    {
      // perror("connect (client)");
      continue;
    }
    break;
  }
  while (1)
  {
    std::unique_lock<std::mutex> lock(m_prio_result_queue_mutex);
    if (m_prio_result_queue.empty())
    {
      continue;
    }
    Data_packet data_packet = m_prio_result_queue.top();
    m_prio_result_queue.pop();
    lock.unlock();
    void *serialized_data_packet = serialize_data_packet(data_packet);
    send(sock, serialized_data_packet,
         data_packet.tensor_size + sizeof(int) * 9, 0);
    // free buffer
    delete[] (char *)serialized_data_packet;
    // printf("frame %d stage %d task_id %d send\n", data_packet.frame_seq,
    //        data_packet.stage, data_packet.task_id);
  }

  close(sock);

  exit(EXIT_SUCCESS);
}
int Worker::m_recv_data_packet()
{
  int sock;
  fd_set active_fd_set, read_fd_set;
  int i;
  struct sockaddr_in clientname;
  size_t size;

  /* Create the socket and set it up to accept connections. */
  sock = make_socket(m_port);
  if (listen(sock, 1) < 0)
  {
    perror("listen");
    exit(EXIT_FAILURE);
  }

  /* Initialize the set of active sockets. */
  FD_ZERO(&active_fd_set);
  FD_SET(sock, &active_fd_set);

  while (1)
  {

    /* Block until input arrives on one or more active sockets. */
    read_fd_set = active_fd_set;
    if (select(FD_SETSIZE, &read_fd_set, NULL, NULL, NULL) < 0)
    {
      perror("select");
      exit(EXIT_FAILURE);
    }
    /* Service all the sockets with input pending. */
    for (i = 0; i < FD_SETSIZE; ++i)
    {
      if (FD_ISSET(i, &read_fd_set))
      {
        if (i == sock)
        {
          /* Connection request on original socket. */
          int new_sock;
          size = sizeof(clientname);
          new_sock =
              accept(sock, (struct sockaddr *)&clientname, (socklen_t *)&size);
          if (new_sock < 0)
          {
            perror("accept");
            exit(EXIT_FAILURE);
          }
          //                    fprintf(stderr,
          //                            "Server: connect from host %s, port
          //                            %hd.\n", inet_ntoa(clientname.sin_addr),
          //                            ntohs(clientname.sin_port));
          FD_SET(new_sock, &active_fd_set);
        }
        else
        {
          /* Data arriving on an already-connected socket. */
          int metadata_buffer[9];
          Data_packet data_packet;
          int valread = read(i, metadata_buffer, sizeof(int) * 9);
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
          recv(i, tensor_buffer, data_packet.tensor_size, MSG_WAITALL);

          std::string s(tensor_buffer, data_packet.tensor_size);
          std::istringstream stream_{s};
          // buffer to stream
          torch::Tensor tensor;
          torch::load(tensor, stream_);

          // create Data_packet
          data_packet.tensor = tensor;

          std::unique_lock<std::mutex> lock(m_prio_task_queue_mutex);
          m_prio_task_queue.push(data_packet);
          // printf("frame %d stage %d task_id %d recv\n",
          // data_packet.frame_seq,
          //        data_packet.stage, data_packet.task_id);
          if (valread < 0)
          {
            close(i);
            FD_CLR(i, &active_fd_set);
          }
        }
      }
    }
  }

  return 0;
}

Master::Master(std::string ip, int port, int stages,
               network last_stage_net, int frames,
               std::vector<partition_parameter> partition_params,
               std::vector<ftp_parameter> ftp_params,
               std::vector<server_address> server_addresses)
    : exit_flag(1)

{
  m_ip = ip;
  m_port = port;
  m_last_stage_net = last_stage_net;
  m_frames = frames;
  m_partition_params = partition_params;
  m_ftp_params = ftp_params;
  m_server_addresses = server_addresses;
  m_stages = stages;
}

LIB_API image_t Master::load_image(std::string image_filename)
{
  char *input = const_cast<char *>(image_filename.c_str());
  image im = load_image_stb(input, 3);

  image_t img;
  img.c = im.c;
  img.data = im.data;
  img.h = im.h;
  img.w = im.w;

  return img;
}

void Master::m_push_image()
{
  std::string image_path = "../dog.jpg";
  int w = m_partition_params[0].in_w;
  int h = m_partition_params[0].in_h;
  int c = m_partition_params[0].in_c;
  for (int i = 0; i < m_frames; ++i)
  {
    auto img = load_image(image_path);
    image im;
    im.c = img.c;
    im.data = img.data;
    im.h = img.h;
    im.w = img.w;
    image sized;
    if (w == im.w && h == im.h)
    {
      sized = make_image(im.w, im.h, im.c);
      memcpy(sized.data, im.data, im.w * im.h * im.c * sizeof(float));
    }
    else
      sized = resize_image(im, w, h);
    // covert image to torch tensor
    c10::IntArrayRef s = {sized.c, sized.h, sized.w};

    torch::Tensor tensor = torch::from_blob(sized.data, s);
    
    tensor = tensor.to(torch::kCUDA);
     
    Data_packet data_packet{i, 0, 0, 0, 0,
                            sized.w, sized.h, sized.c, 0, tensor};
    // std::cout << *sized.data << std::endl;
    // push image to queue
    std::unique_lock<std::mutex> lock(m_prio_image_queue_mutex);
    m_prio_image_queue.push(data_packet);
    lock.unlock();
    free(im.data);
    free(sized.data);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    frame_time_point[i] = {t1, t1};
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
  printf("push image exit\n");
}

void Master::m_pritition_image()
{

  while (exit_flag)
  {
    std::unique_lock<std::mutex> lock(m_prio_image_queue_mutex);
    if (m_prio_image_queue.empty())
    {
      continue;
    }
    // queue is not empty
    Data_packet data_packet = m_prio_image_queue.top();
    m_prio_image_queue.pop();
    lock.unlock();
    
    int frame_seq = data_packet.frame_seq;
    int stage = data_packet.stage;

    for (int i = 0; i < m_ftp_params[stage].partitions_h; ++i)
    {
      for (int j = 0; j < m_ftp_params[stage].partitions_w; ++j)
      {
        int task_id = m_ftp_params[stage].task_ids[i][j];
        int dw1 = m_ftp_params[stage].input_tiles[task_id][0].w1;
        int dw2 = m_ftp_params[stage].input_tiles[task_id][0].w2;
        int dh1 = m_ftp_params[stage].input_tiles[task_id][0].h1;
        int dh2 = m_ftp_params[stage].input_tiles[task_id][0].h2;
        int c = m_ftp_params[stage].input_tiles[task_id][0].c;
        // crop
        torch::Tensor partition =
            data_packet.tensor
                .index({at::indexing::Slice(0, c),
                        at::indexing::Slice(dh1, dh2 + 1),
                        at::indexing::Slice(dw1, dw2 + 1)})
                .clone();
        // flip

        // assert(partition.sizes()[0] == c);
        // assert(partition.sizes()[1] == dh2 - dh1 + 1);
        // assert(partition.sizes()[2] == dw2 - dw1 + 1);
        torch::Tensor fliped_partition;
        switch (task_id)
        {
        case 0:
          fliped_partition = partition;
          break;
        case 1:
          // flip lr
          fliped_partition = partition.flip(2);
          break;
        case 2:
          // flip ud
          fliped_partition = partition.flip(1);
          break;
        case 3:
          fliped_partition = partition.flip({1, 2});
          break;
        }

        Data_packet new_data_packet{frame_seq,
                                    task_id,
                                    stage,
                                    m_partition_params[stage].from,
                                    m_partition_params[stage].to,
                                    dw2 - dw1 + 1,
                                    dh2 - dh1 + 1,
                                    c,
                                    0,
                                    fliped_partition};
        std::unique_lock<std::mutex> lock1(m_prio_task_queue_mutex);
        m_prio_task_queue.push(new_data_packet);

        
      }
    }
  }
  printf("partition_image thread end\n");
}

void Master::m_merge_partitions()
{

  while (exit_flag)
  {
    int counts = 0;
    std::unique_lock<std::mutex> lock(m_prio_partition_inference_result_mutex);
    if (m_prio_partition_inference_result_queue.empty())
    {
      continue;
    }
    // auto start = std::chrono::high_resolution_clock::now();
    Data_packet data_packet = m_prio_partition_inference_result_queue.front();
    lock.unlock();

    int frame_seq = data_packet.frame_seq;
    int task_id = data_packet.task_id;
    int stage = data_packet.stage;
    int from = data_packet.from;
    int to = data_packet.to;
    // create merged image
    c10::IntArrayRef s = {m_partition_params[stage].out_c, m_partition_params[stage].out_h,
                          m_partition_params[stage].out_w};
    torch::Tensor merged =
        torch::rand(s, torch::TensorOptions().device(torch::kCUDA));
    // assert(merged.sizes()[0] == m_net.layers[to].out_c);
    // assert(merged.sizes()[1] == m_net.layers[to].out_h);
    // assert(merged.sizes()[2] == m_net.layers[to].out_w);

    while (counts < m_partition_params[stage].partitions)
    {
      std::unique_lock<std::mutex> lock1(
          m_prio_partition_inference_result_mutex);
      if (m_prio_partition_inference_result_queue.empty())
      {
        continue;
      }
      Data_packet data_packet = m_prio_partition_inference_result_queue.front();
      if (counts != 0 && data_packet.frame_seq != frame_seq ||
          data_packet.stage != stage)
      {

        m_prio_partition_inference_result_queue.pop();
        m_prio_partition_inference_result_queue.push(data_packet);
        continue;
      }

      m_prio_partition_inference_result_queue.pop();
      lock1.unlock();
      ++counts;
      int frame_seq = data_packet.frame_seq;
      int task_id = data_packet.task_id;
      int stage = data_packet.stage;
      int from = data_packet.from;
      int to = data_packet.to;
      torch::Tensor partition = data_packet.tensor;

      // assert(partition.sizes()[0] ==
      // m_sub_nets[stage][task_id].layers[14].out_c);
      // assert(partition.sizes()[1] ==
      // m_sub_nets[stage][task_id].layers[14].out_h);
      // assert(partition.sizes()[2] ==
      // m_sub_nets[stage][task_id].layers[14].out_w);
      int fused_layers = m_ftp_params[stage].fused_layers;
      int dw1 = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].w1;
      int dw2 = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].w2;
      int dh1 = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].h1;
      int dh2 = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].h2;
      int w = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].w;
      int h = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].h;
      int c = m_ftp_params[stage].output_tiles[task_id][fused_layers - 1].c;
      // crop padding
      // create merged image

      torch::Tensor cropped_partition =
          partition
              .index({at::indexing::Slice(0, c), at::indexing::Slice(0, h),
                      at::indexing::Slice(0, w)})
              .clone();
      // flip data
      // assert(cropped_partition.sizes()[0] == c);
      // assert(cropped_partition.sizes()[1] == h);
      // assert(cropped_partition.sizes()[2] == w);
      torch::Tensor fliped_cropped_partition;
      switch (task_id)
      {
      case 0:
        fliped_cropped_partition = cropped_partition;
        break;
      case 1:
        // flip lr
        fliped_cropped_partition = cropped_partition.flip(2);
        break;
      case 2:
        // flip ud
        fliped_cropped_partition = cropped_partition.flip(1);
        break;
      case 3:
        fliped_cropped_partition = cropped_partition.flip({1, 2});
        break;
      }
      // assert(cropped_partition.sizes()[0] == c);
      // assert(cropped_partition.sizes()[1] == dh2 - dh1 + 1);
      // assert(cropped_partition.sizes()[2] == dw2 - dw1 + 1);

      // stitch partition to original image
      merged.index({at::indexing::Slice(0, c),
                    {at::indexing::Slice(dh1, dh1 + h)},
                    {at::indexing::Slice(dw1, dw1 + w)}}) =
          fliped_cropped_partition;
    }

    Data_packet new_data_packet{frame_seq,
                                task_id,
                                stage + 1,
                                m_partition_params[stage + 1].from,
                                m_partition_params[stage + 1].to,
                                m_partition_params[stage].out_w,
                                m_partition_params[stage].out_h,
                                m_partition_params[stage].out_c,
                                0,
                                merged};
    if (stage + 1 < m_stages - 1)
    {
      std::unique_lock<std::mutex> lock2(m_prio_image_queue_mutex);
      m_prio_image_queue.push(new_data_packet);
    }
    else
    {
      new_data_packet.tensor = new_data_packet.tensor.to(torch::kCPU);
      std::unique_lock<std::mutex> lock2(m_prio_merged_result_mutex);
      m_prio_merged_result_queue.push(new_data_packet);
    }
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration =
    //     std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // std::cout << "merged function: " << duration.count()
    //           << " milliseconds" << std::endl;
  }
  printf("merge partition thread end\n");
}
// handle mutilple client
int Master::m_recv_data_packet()
{

  int sock;
  fd_set active_fd_set, read_fd_set;
  int i;
  struct sockaddr_in clientname;
  size_t size;

  /* Create the socket and set it up to accept connections. */
  sock = make_socket(m_port);
  if (listen(sock, 1) < 0)
  {
    perror("listen");
    exit(EXIT_FAILURE);
  }

  /* Initialize the set of active sockets. */
  FD_ZERO(&active_fd_set);
  FD_SET(sock, &active_fd_set);
  auto start = std::chrono::high_resolution_clock::now();
  while (1)
  {
    /* Block until input arrives on one or more active sockets. */
    read_fd_set = active_fd_set;
    if (select(FD_SETSIZE, &read_fd_set, NULL, NULL, NULL) < 0)
    {
      perror("select");
      exit(EXIT_FAILURE);
    }
    /* Service all the sockets with input pending. */
    for (i = 0; i < FD_SETSIZE; ++i)
    {

      if (FD_ISSET(i, &read_fd_set))
      {
        if (i == sock)
        {
          /* Connection request on original socket. */
          int new_sock = 0;
          size = sizeof(clientname);
          new_sock =
              accept(sock, (struct sockaddr *)&clientname, (socklen_t *)&size);
          if (new_sock < 0)
          {
            perror("accept");
            exit(EXIT_FAILURE);
          }
          //                    fprintf(stderr,
          //                            "Server: connect from host %s, port
          //                            %hd.\n", inet_ntoa(clientname.sin_addr),
          //                            ntohs(clientname.sin_port));
          FD_SET(new_sock, &active_fd_set);
        }
        else
        {
          /* Data arriving on an already-connected socket. */
          int metadata_buffer[9];
          Data_packet data_packet;
          int valread = read(i, metadata_buffer, sizeof(int) * 9);
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
          recv(i, tensor_buffer, data_packet.tensor_size, MSG_WAITALL);
          std::string s(tensor_buffer, data_packet.tensor_size);
          std::istringstream stream_{s};
          // buffer to stream
          torch::Tensor tensor;
          torch::load(tensor, stream_);
          // create Data_packet
          data_packet.tensor = tensor;

          // push to queue
          std::unique_lock<std::mutex> lock(
              m_prio_partition_inference_result_mutex);
          m_prio_partition_inference_result_queue.push(data_packet);
          // printf("frame %d stage %d task_id %d recv\n",
          // data_packet.frame_seq,
          //        data_packet.stage, data_packet.task_id);
          auto stop = std::chrono::high_resolution_clock::now();
          auto duration =
              std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

          // std::cout << "recv: " << duration.count()
          //           << " milliseconds"
          //           << " " << data_packet.frame_seq << std::endl;
          start = stop;
          if (valread < 0)
          {
            close(i);
            FD_CLR(i, &active_fd_set);
          }
        }
      }
    }
  }
  printf("recv thread end\n");
  return 0;
}

int Master::m_send_data_packet()
{

  // create multiple filedes
  std::vector<int> socks(m_server_addresses.size(), 0);

  // connections
  for (int i = 0; i < m_server_addresses.size(); ++i)
  {
    struct sockaddr_in servername;
    /* Create the socket. */
    socks[i] = socket(AF_INET, SOCK_STREAM, 0);
    init_sockaddr(&servername, m_server_addresses[i].ip.c_str(),
                  m_server_addresses[i].port);
    if (socks[i] < 0)
    {
      perror("socket (client)");
      exit(EXIT_FAILURE);
    }
    /* Connect to the server. */
    while (1)
    {
      if (0 > connect(socks[i], (struct sockaddr *)&servername,
                      sizeof(servername)))
      {
        // perror("connect (client)");
        continue;
      }
      break;
    }
  }
  int target_sock = 0;

  while (exit_flag)
  {

    std::unique_lock<std::mutex> lock(m_prio_task_queue_mutex);
    if (m_prio_task_queue.empty())
    {
      continue;
    }
    Data_packet data_packet = m_prio_task_queue.top();
    m_prio_task_queue.pop();
    data_packet.tensor = data_packet.tensor.to(torch::kCPU);
    void *serialized_data_packet = serialize_data_packet(data_packet);
    send(socks[target_sock % m_server_addresses.size()], serialized_data_packet,
         data_packet.tensor_size + sizeof(int) * 9, 0);
    // printf("frame %d stage %d task_id %d send\n", data_packet.frame_seq,
    //        data_packet.stage, data_packet.task_id);
    delete[] (char *)serialized_data_packet;
    target_sock++;
  }

  printf("send thread end\n");

  return 0;
}

inline static void *serialize_data_packet(Data_packet &data_packet)
{
  std::ostringstream stream;
  torch::save(data_packet.tensor, stream);
  const std::string str = stream.str();
  const int length = str.length();
  data_packet.tensor_size = length;

  char *buffer = new char[sizeof(int) * 9 + length];

  std::memcpy(buffer, &data_packet, sizeof(int) * 9);
  char *p = buffer + sizeof(int) * 9;
  std::memcpy(p, str.data(), length);
  return buffer;
}
// inference final stage
void Master::m_inference()
{
  int completed_frames = 0;
  while (exit_flag)
  {
    std::unique_lock<std::mutex> lock(m_prio_merged_result_mutex);
    if (m_prio_merged_result_queue.empty())
    {
      continue;
    }

    Data_packet data_packet = m_prio_merged_result_queue.top();
    m_prio_merged_result_queue.pop();
    lock.unlock();

    float *X = data_packet.tensor.data_ptr<float>();
    // layer merged_layer =
    //     m_net.layers[m_partition_params[data_packet.stage - 1].to];
    // for (int i = 0; i < data_packet.h; ++i) {
    //   for (int j = 0; j < data_packet.w; ++j) {
    //     std::string a_ = std::to_string(
    //         *(m_net.layers[m_partition_params[data_packet.stage - 1].to - 1]
    //               .output +
    //           i * data_packet.w + j));
    //     std::cout << a_ << " ";
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << std::endl << std::endl;
    // for (int i = 0; i < data_packet.h; ++i) {
    //   for (int j = 0; j < data_packet.w; ++j) {
    //     std::string a_ = std::to_string(*(
    //         m_net.layers[m_partition_params[data_packet.stage - 1].to].output
    //         + i * data_packet.w + j));
    //     std::cout << a_ << " ";
    //   }
    //   std::cout << std::endl;
    // }

    float *out = network_predict(m_last_stage_net, X);

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    frame_time_point[data_packet.frame_seq].second = t2;

    // layer l = m_last_stage_net.layers[m_last_stage_net.n - 1];
    // assert(data_packet.tensor.sizes()[0] == m_last_stage_net.c);
    // assert(data_packet.tensor.sizes()[1] == m_last_stage_net.h);
    // assert(data_packet.tensor.sizes()[2] == m_last_stage_net.w);

    // for (int i = 0; i < l.out_w * l.out_h * l.out_c; ++i) {
    //   std::string a = std::to_string(*(m_net.layers[m_net.n - 1].output +
    //   i)); std::string b = std::to_string(*(out + i)); std::cout << a << " "
    //   + std::to_string(i) + " " << b << std::endl; assert(a.substr(0, 4) ==
    //   b.substr(0, 4));
    // }

    // auto img = load_image("./data/dog.jpg");
    // image im;
    // im.c = img.c;
    // im.data = img.data;
    // im.h = img.h;
    // im.w = img.w;

    // image sized;

    // if (m_net.w == im.w && m_net.h == im.h) {
    //   sized = make_image(im.w, im.h, im.c);
    //   memcpy(sized.data, im.data, im.w * im.h * im.c * sizeof(float));
    // } else
    //   sized = resize_image(im, m_net.w, m_net.h);

    // l = m_last_stage_net.layers[m_last_stage_net.n - 1];
    // float thresh = 0.5;
    // int nboxes = 0;
    // int letterbox = 0;
    // float hier_thresh = 0.5;
    // float nms = .4;
    // detection *dets = get_network_boxes(&m_last_stage_net, im.w, im.h,
    // thresh,
    //                                     hier_thresh, 0, 1, &nboxes,
    //                                     letterbox);
    // if (nms)
    //   do_nms_sort(dets, nboxes, l.classes, nms);

    // std::vector<bbox_t> bbox_vec;

    // for (int i = 0; i < nboxes; ++i) {
    //   box b = dets[i].bbox;
    //   int const obj_id = max_index(dets[i].prob, l.classes);
    //   float const prob = dets[i].prob[obj_id];

    //   if (prob > thresh) {
    //     bbox_t bbox;
    //     bbox.x = std::max((double)0, (b.x - b.w / 2.) * im.w);
    //     bbox.y = std::max((double)0, (b.y - b.h / 2.) * im.h);
    //     bbox.w = b.w * im.w;
    //     bbox.h = b.h * im.h;
    //     bbox.obj_id = obj_id;
    //     bbox.prob = prob;
    //     bbox.track_id = 0;
    //     bbox.frames_counter = 0;
    //     bbox.x_3d = NAN;
    //     bbox.y_3d = NAN;
    //     bbox.z_3d = NAN;

    //     bbox_vec.push_back(bbox);
    //   }
    // }

    // free_detections(dets, nboxes);
    // if (sized.data)
    //   free(sized.data);

    // // #ifdef GPU
    // //     if (cur_gpu_id != old_gpu_index)
    // //         cudaSetDevice(old_gpu_index);
    // // #endif
    // std::vector<bbox_t> result_vec = bbox_vec;
    // free(img.data);
    // auto obj_names = objects_names_from_file("./data/coco.names");
    // show_console_result(result_vec, obj_names, 0);
    ++completed_frames;
    if (completed_frames == m_frames)
    {
      exit_flag = 0;
      printf("master exit\n");
      break;
    }
  }
}