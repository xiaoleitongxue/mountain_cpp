#ifndef PARTITION_MODEL_HPP
#define PARTITION_MODEL_HPP
#include "darknet.h"
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
#include "data.h"
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
}

// #include "stb_image.h"

typedef struct server_address {
  std::string ip;
  int port;
} server_address;

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
  def_ftp_para(int partitions_w_, int partitions_h_, int from, int to)
      : partitions_w(partitions_w_), partitions_h(partitions_h_),
        fused_layers(to - from + 1),
        task_ids(partitions_h_, std::vector<int>(partitions_w_)),
        partitions(partitions_h_ * partitions_w_),
        input_tiles(partitions, std::vector<tile_region>(to - from + 1)),
        output_tiles(partitions, std::vector<tile_region>(to - from + 1)) {}
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