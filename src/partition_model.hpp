#ifndef PARTITION_MODEL_H
#define PARTITION_MODEL_H
#include "darknet.h"
#include "yolo_v2_class.hpp"
#include <system_error>
#include <vector>

#include "network.h"

extern "C" {
#include "activation_layer.h"
#include "activations.h"
#include "assert.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "conv_lstm_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gaussian_yolo_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "lstm_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "reorg_old_layer.h"
#include "representation_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "sam_layer.h"
#include "scale_channels_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "upsample_layer.h"
#include "utils.h"
#include "version.h"
#include "yolo_layer.h"
// #include "stb_image.h"
}

typedef struct {
  char *type;
  list *options;
} section;
typedef struct size_params {
  int batch;
  int inputs;
  int h;
  int w;
  int c;
  int index;
  int time_steps;
  int train;
  network net;
} size_params;
extern "C" LIB_API {
  void empty_func(dropout_layer l, network_state state);
  list *read_cfg(char *filename);
  LAYER_TYPE string_to_layer_type(char *type);
  void free_section(section * s);
  void parse_data(char *data, float *a, int n);
  struct size_params;
  local_layer parse_local(list * options, size_params params);
  convolutional_layer parse_convolutional(list * options, size_params params);
  layer parse_crnn(list * options, size_params params);
  layer parse_rnn(list * options, size_params params);
  layer parse_gru(list * options, size_params params);
  layer parse_lstm(list * options, size_params params);
  layer parse_conv_lstm(list * options, size_params params);
  layer parse_history(list * options, size_params params);
  connected_layer parse_connected(list * options, size_params params);
  softmax_layer parse_softmax(list * options, size_params params);
  contrastive_layer parse_contrastive(list * options, size_params params);
  int *parse_yolo_mask(char *a, int *num);
  float *get_classes_multipliers(char *cpc, const int classes,
                                 const float max_delta);
  layer parse_yolo(list * options, size_params params);
  int *parse_gaussian_yolo_mask(char *a, int *num);
  layer parse_gaussian_yolo(list * options, size_params params);
  layer parse_region(list * options, size_params params);
  detection_layer parse_detection(list * options, size_params params);
  cost_layer parse_cost(list * options, size_params params);
  crop_layer parse_crop(list * options, size_params params);
  layer parse_reorg(list * options, size_params params);
  layer parse_reorg_old(list * options, size_params params);
  maxpool_layer parse_local_avgpool(list * options, size_params params);
  maxpool_layer parse_maxpool(list * options, size_params params);
  avgpool_layer parse_avgpool(list * options, size_params params);
  dropout_layer parse_dropout(list * options, size_params params);
  layer parse_normalization(list * options, size_params params);
  layer parse_batchnorm(list * options, size_params params);
  layer parse_shortcut(list * options, size_params params, network net);
  layer parse_scale_channels(list * options, size_params params, network net);
  layer parse_sam(list * options, size_params params, network net);
  layer parse_implicit(list * options, size_params params, network net);
  layer parse_activation(list * options, size_params params);
  layer parse_upsample(list * options, size_params params, network net);
  route_layer parse_route(list * options, size_params params);
  learning_rate_policy get_policy(char *s);
  void parse_net_options(list * options, network * net);
  int is_network(section * s);
  void set_train_only_bn(network net);
  list *read_cfg(char *filename);
  void save_convolutional_weights_binary(layer l, FILE * fp);
  void save_shortcut_weights(layer l, FILE * fp);
  void save_implicit_weights(layer l, FILE * fp);
  void save_convolutional_weights(layer l, FILE * fp);
  void save_convolutional_weights_ema(layer l, FILE * fp);
  void save_batchnorm_weights(layer l, FILE * fp);
  void save_connected_weights(layer l, FILE * fp);
  void transpose_matrix(float *a, int rows, int cols);
  void load_connected_weights(layer l, FILE * fp, int transpose);
  void load_batchnorm_weights(layer l, FILE * fp);
  void load_convolutional_weights_binary(layer l, FILE * fp);
  void load_convolutional_weights(layer l, FILE * fp);
  void load_shortcut_weights(layer l, FILE * fp);
  void load_implicit_weights(layer l, FILE * fp);
}

typedef struct partition_parameter {
  int partition_w;
  int partition_h;
  int partitions;
  int from;
  int to;
} partition_parameter;

typedef struct partition_range {
  int w1;
  int h1;
  int w2;
  int h2;
  int h;
  int w;
  int c; /*Channel number*/
} tile_region;

typedef struct def_ftp_para {
  int partitions;
  int partitions_w;
  int partitions_h;
  int fused_layers;
  std::vector<std::vector<int>> task_ids;
  std::vector<std::vector<tile_region>> input_tiles;
  std::vector<std::vector<tile_region>> output_tiles;
  def_ftp_para(int partitions_w_, int partitions_h_, int from, int to) {
    partitions_w = partitions_w_;
    partitions_h = partitions_h_;
    fused_layers = to - from + 1;
    partitions = partitions_h_ * partitions_w_;
    task_ids = std::vector(partitions_h_, std::vector<int>(partitions_w_));
    input_tiles =
        std::vector(partitions, std::vector<tile_region>(to - from + 1));
    output_tiles =
        std::vector(partitions, std::vector<tile_region>(to - from + 1));
  }
} ftp_parameter;

std::vector<ftp_parameter>
perform_ftp(std::vector<partition_parameter> partition_para, int stages,
            network net);
std::vector<std::vector<network>>
generate_sub_network(char *cfg_file, std::vector<ftp_parameter> ftp_params,
                     int stages,
                     std::vector<partition_parameter> partition_params);
network parse_network_cfg_custom_whc(char *filename,
                                     partition_parameter partition_param,
                                     ftp_parameter ftp_param, int task_id,
                                     int stages, int stage, int batch,
                                     int time_steps);

void load_weights_upto_subnet(network net, network &sub_net, char *filename,
                              int start, int cutoff, int start_layer,
                              int end_layer);
void load_sub_nets_weights(std::vector<std::vector<network>> sub_nets,
                           char *filename, char *weights, int stages,
                           std::vector<partition_parameter> partition_params);
#endif