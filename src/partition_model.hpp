#ifndef PARTITION_MODEL_HPP
#define PARTITION_MODEL_HPP
#include <darknet_include.hpp>

typedef struct server_address {
  std::string ip;
  int port;
} server_address;




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

void load_weights_upto_subnet(network *net, network *sub_net, char *filename,
                              int cutoff, int start_layer,
                              int end_layer);
void load_sub_nets_weights(std::vector<std::vector<network>> &sub_nets,
                           char *filename, char *weights, int stages,
                           std::vector<partition_parameter> partition_params);
#endif