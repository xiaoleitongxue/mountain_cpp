#ifndef DATA_BLOB_HPP
#define DATA_BLOB_HPP
#include <ATen/core/TensorBody.h>
#include <sstream>
#include <torch/serialize.h>
#include <torch/torch.h>
typedef struct Data_packet {
  int frame_seq;
  int task_id;
  int stage;
  int from; // original network layer index
  int to;   // original network layer index
  // dims of tensor
  int w;
  int h;
  int c;
  int tensor_size;
  torch::Tensor tensor;
} Data_packet;


#endif