#include <inference_helper.hpp>
#include "darknet.h"
float *fliplr_feature_map(float *input, int w, int h, int c) {
  for (int i = 0; i < c; ++i) {
    // 行内翻转
    for (int j = 0; j < h; ++j) {
      int start = 0 + w * j + w * h * i;
      int end = w + w * j + w * h * i - 1;
      while (start < end) {
        swap(input, start, end);
        start++;
        end--;
      }
    }
  }
  return input;
}

float *flipud_feature_map(float *input, int w, int h, int c) {
  for (int i = 0; i < c; ++i) {
    // 行内翻转
    for (int j = 0; j < h; ++j) {
      int start = w * j + w * h * i;
      int end = w + w * j + w * h * i - 1;
      while (start < end) {
        swap(input, start, end);
        start++;
        end--;
      }
    }
    int start = w * h * i;
    int end = w * h * i + w * h - 1;
    while (start < end) {
      swap(input, start, end);
      start++;
      end--;
    }
  }
  return input;
}

float *flipx_feature_map(float *input, int w, int h, int c) {
  for (int i = 0; i < c; ++i) {
    int offset = i * w * h;
    int start = offset;
    int end = offset + w * h - 1;
    while (start < end) {
      swap(input, start, end);
      start++;
      end--;
    }
  }
  return input;
}

float *flip_feature_map(int type, float *input, int w, int h, int c) {
  switch (type) {
  case 1:
    fliplr_feature_map(input, w, h, c);
    break;
  case 2:
    flipud_feature_map(input, w, h, c);
    break;
  case 3:
    flipx_feature_map(input, w, h, c);
    break;
  default:
    break;
  }
  return input;
}
void flip_sub_nets_weghts(std::vector<std::vector<network>> sub_nets, int stages,
                          std::vector<partition_parameter> partition_params,
                          std::vector<ftp_parameter> ftp_params) {
  for (int i = 0; i < stages; ++i) {
    for (int j = 0; j < partition_params[i].partitions; ++j) {
      network sub_net = sub_nets[i][j];
      for (int k = 0; k < sub_net.n; ++k) {
        layer l = sub_net.layers[k];
        if (l.type == CONVOLUTIONAL) {
          for (int p = 0; p < l.c / l.groups; ++p) {
            int offset = p * l.size * l.size * l.n;
            flip_feature_map(j, l.weights + offset, l.size, l.size, l.n);
          }
#ifdef GPU
          if (gpu_index >= 0) {
            push_convolutional_layer(l);
          }
#endif
        }
      }
    }
  }
}

void swap(float *input, int start, int end) {
  float temp = input[start];
  input[start] = input[end];
  input[end] = temp;
}