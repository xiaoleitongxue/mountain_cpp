#ifndef INFERENCE_HELPER_HPP
#define INFERENCE_HELPER_HPP

#include <partition_model.hpp>
#include <yolo_v2_class.hpp>
float *flip_feature_map(int type, float *input, int w, int h, int c);
float *fliplr_feature_map(float *input, int w, int h, int c);
float *flipud_feature_map(float *input, int w, int h, int c);
float *flipx_feature_map(float *input, int w, int h, int c);
void flip_sub_nets_weghts(std::vector<std::vector<network>> sub_nets,
                          int stages,
                          std::vector<partition_parameter> partition_params,
                          std::vector<ftp_parameter> ftp_params);
void swap(float *input, int start, int end);

#endif