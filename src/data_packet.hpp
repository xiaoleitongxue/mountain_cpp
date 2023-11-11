#ifndef DATA_PACKET_HPP
#define DATA_PACKET_HPP
#include <torch/torch.h>
#include <torch/serialize/input-archive.h>
typedef struct Data_packet {
  int frame_seq;
  int task_id;
  int stage;
  int from; // original network layer index
  int to;   // original network layer index
  int w;
  int h;
  int c;
  int tensor_size;
  torch::Tensor tensor;
} Data_packet;



// typedef struct Data_packet Data_packet;


#endif