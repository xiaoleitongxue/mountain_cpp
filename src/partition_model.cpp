#include "parser.h"
#include <cstdlib>
#include <cstring>
#include <darknet.h>
#include <iterator>
#include <partition_model.hpp>
#include <string>
#include <vector>

void grid(network net, ftp_parameter &ftp_para, int partition_w,
          int partition_h, int from, int to)
{

  int w = net.layers[to].out_w;
  int h = net.layers[to].out_h;
  int c = net.layers[to].out_c;
  int stride_w = ceil(((float)w) / ((float)partition_w));
  int start_w = 0;
  int end_w = stride_w - 1;
  int stride_h = ceil(((float)h) / ((float)partition_h));
  int start_h = 0;
  int end_h = stride_h - 1;

  for (int i = 0; i < partition_h; i++)
  {
    start_w = 0;
    end_w = stride_w - 1;
    for (int j = 0; j < partition_w; ++j)
    {
      int task_id = ftp_para.task_ids[i][j];
      ftp_para.output_tiles[task_id][to - from].w1 = start_w;
      ftp_para.output_tiles[task_id][to - from].w2 = end_w;
      ftp_para.output_tiles[task_id][to - from].h1 = start_h;
      ftp_para.output_tiles[task_id][to - from].h2 = end_h;
      ftp_para.output_tiles[task_id][to - from].h = end_h - start_h + 1;
      ftp_para.output_tiles[task_id][to - from].w = end_w - start_w + 1;
      ftp_para.output_tiles[task_id][to - from].c = c;
      start_w = end_w + 1;
      if (j == (partition_w - 2))
      {
        end_w = w - 1;
      }
      else
      {
        end_w = end_w + stride_w;
      }
    }
    start_h = end_h + 1;
    if (i == (partition_h - 2))
    {
      end_h = h - 1;
    }
    else
    {
      end_h = end_h + stride_h;
    }
  }
}

tile_region traversal(tile_region &output, int i, network net)
{
  tile_region input;
  layer l = net.layers[i];
  int stride = l.stride;
  int filter_size = l.size;
  int w = l.w;
  int h = l.h;
  int c = l.c;
  if (l.type == CONVOLUTIONAL)
  {
    input.w1 = (output.w1 * stride - filter_size / 2) > 0
                   ? (output.w1 * stride - filter_size / 2)
                   : 0;
    input.w2 = (output.w2 * stride + filter_size / 2) < (w - 1)
                   ? (output.w2 * stride + filter_size / 2)
                   : (w - 1);
    input.h1 = (output.h1 * stride - filter_size / 2) > 0
                   ? (output.h1 * stride - filter_size / 2)
                   : 0;
    input.h2 = (output.h2 * stride + filter_size / 2) < (h - 1)
                   ? (output.h2 * stride + filter_size / 2)
                   : (h - 1);
  }
  else if (net.layers[i].type == MAXPOOL)
  {
    if (stride == filter_size)
    {
      input.w1 = output.w1 * stride;
      input.w2 = output.w2 * stride + filter_size - 1;
      input.h1 = output.h1 * stride;
      input.h2 = output.h2 * stride + filter_size - 1;
    }
    else
    {
      input.w1 = (output.w1 * stride) > 0 ? (output.w1 * stride) : 0;
      input.w2 = (output.w2 * stride + filter_size - 1) < (w - 1)
                     ? (output.w2 * stride + filter_size - 1)
                     : (w - 1);
      input.h1 =
          (output.h1 * stride) > 0 ? (output.h1 * stride) : 0;
      input.h2 = (output.h2 * stride + filter_size - 1) < (h - 1)
                     ? ((output.h2 * stride + filter_size - 1))
                     : (h - 1);
    }
  }
  input.w = input.w2 - input.w1 + 1;
  input.h = input.h2 - input.h1 + 1;
  input.c = c;
  return input;
}

std::vector<ftp_parameter>
perform_ftp(std::vector<partition_parameter> partition_params, int stages,
            network net)
{
  std::vector<ftp_parameter> ftp_params;
  for (int i = 0; i < stages; ++i)
  {
    int partition_w = partition_params[i].partition_w;
    int partition_h = partition_params[i].partition_h;
    int from = partition_params[i].from;
    int to = partition_params[i].to;
    int partitions = partition_params[i].partitions;

    ftp_params.push_back(ftp_parameter{partition_w, partition_h, from, to});

    // ftp_params[i].partitions = partition_w * partition_h;
    // ftp_params[i].partitions_h = partition_h;
    // ftp_params[i].partitions_w = partition_w;
    // ftp_params[i].fused_layers = to - from + 1;

    // ftp_params[i].task_ids.resize(partition_h, std::vector<int>(partition_w,
    // 0));
    int id = 0;
    for (int j = 0; j < partition_h; ++j)
    {
      for (int k = 0; k < partition_w; ++k)
      {
        ftp_params[i].task_ids[j][k] = id;
        ++id;
      }
    }
    if (partitions > 1)
    {
      // resize ftp
      //  ftp_params[i].input_tiles.resize(partition_h * partition_w,
      //  std::vector<tile_region>(to - from + 1));
      //  ftp_params[i].output_tiles.resize(partition_h * partition_w,
      //  std::vector<tile_region>(to - from + 1));
      grid(net, ftp_params[i], partition_w, partition_h, from, to);
      for (int k = 0; k < ftp_params[i].partitions_h; k++)
      {
        for (int p = 0; p < ftp_params[i].partitions_w; p++)
        {
          for (int q = to - from; q >= 0; --q)
          {
            ftp_params[i].input_tiles[ftp_params[i].task_ids[k][p]][q] =
                traversal(
                    ftp_params[i].output_tiles[ftp_params[i].task_ids[k][p]][q],
                    from + q, net);
            if (q > 0)
              ftp_params[i].output_tiles[ftp_params[i].task_ids[k][p]][q - 1] =
                  ftp_params[i].input_tiles[ftp_params[i].task_ids[k][p]][q];
          }
        }
      }
    }
    else
    {
      // final stage nets
      for (int r = from; r <= to; r++)
      {
        ftp_params[i].input_tiles[0][r - from].w = net.layers[r].w;
        ftp_params[i].input_tiles[0][r - from].h = net.layers[r].h;
        ftp_params[i].input_tiles[0][r - from].c = net.layers[r].c;
        ftp_params[i].input_tiles[0][r - from].h1 = 0;
        ftp_params[i].input_tiles[0][r - from].h2 = net.layers[r].h - 1;
        ftp_params[i].input_tiles[0][r - from].w1 = 0;
        ftp_params[i].input_tiles[0][r - from].w2 = net.layers[r].w - 1;

        ftp_params[i].output_tiles[0][r - from].w = net.layers[r].out_w;
        ftp_params[i].output_tiles[0][r - from].h = net.layers[r].out_h;
        ftp_params[i].output_tiles[0][r - from].c = net.layers[r].out_c;
        ftp_params[i].output_tiles[0][r - from].h1 = 0;
        ftp_params[i].output_tiles[0][r - from].h2 = net.layers[r].out_h - 1;
        ftp_params[i].output_tiles[0][r - from].w1 = 0;
        ftp_params[i].output_tiles[0][r - from].w2 = net.layers[r].out_w - 1;
      }
    }
  }

  // final stage is not needs to partiton, just copy from original network

  return ftp_params;
}

network parse_network_cfg_custom_whc(char *filename,
                                     partition_parameter partition_param,
                                     ftp_parameter ftp_param, int task_id,
                                     int batch,
                                     int time_steps)
{
  network net;
  return net;
}

void load_sub_nets_weights(network net, std::vector<std::vector<network>> &sub_nets,
                           char *cfg_file, char *weights, int stages,
                           std::vector<partition_parameter> partition_params)
{
  
}

void load_weights_upto_subnet(network *net, network *sub_net, char *filename,
                              int cutoff, int start_layer, int end_layer)
{

}