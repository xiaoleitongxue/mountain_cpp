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
  list *sections = read_cfg(filename);
  node *n = sections->front;
  if (!n)
    error("Config file has no sections", DARKNET_LOC);
  network net = make_network(partition_param.to - partition_param.from + 1);
  net.gpu_index = gpu_index;
  size_params params;

  if (batch > 0)
    params.train = 0; // allocates memory for Inference only
  else
    params.train = 1; // allocates memory for Inference & Training

  section *s = (section *)n->val;
  list *options = s->options;
  if (!is_network(s))
    error("First section must be [net] or [network]", DARKNET_LOC);
  parse_net_options(options, &net);

#ifdef GPU
  printf("net.optimized_memory = %d \n", net.optimized_memory);
  if (net.optimized_memory >= 2 && params.train)
  {
    pre_allocate_pinned_memory(
        (size_t)1024 * 1024 * 1024 *
        8); // pre-allocate 8 GB CPU-RAM for pinned memory
  }
#endif // GPU

  net.w = ftp_param.input_tiles[task_id][0].w;
  net.h = ftp_param.input_tiles[task_id][0].h;
  net.c = ftp_param.input_tiles[task_id][0].c;
  net.inputs = net.w * net.h * net.c;
  net.max_crop = net.w * 2;
  net.min_crop = net.w;
  params.h = net.h;
  params.w = net.w;
  params.c = net.c;
  params.inputs = net.inputs;
  if (batch > 0)
    net.batch = batch;
  if (time_steps > 0)
    net.time_steps = time_steps;
  if (net.batch < 1)
    net.batch = 1;
  if (net.time_steps < 1)
    net.time_steps = 1;
  if (net.batch < net.time_steps)
    net.batch = net.time_steps;
  params.batch = net.batch;
  params.time_steps = net.time_steps;
  params.net = net;
  printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n",
         net.batch, net.batch * net.subdivisions, net.time_steps, params.train);

  int last_stop_backward = -1;
  int avg_outputs = 0;
  int avg_counter = 0;
  float bflops = 0;
  size_t workspace_size = 0;
  size_t max_inputs = 0;
  size_t max_outputs = 0;
  int receptive_w = 1, receptive_h = 1;
  int receptive_w_scale = 1, receptive_h_scale = 1;
  const int show_receptive_field =
      option_find_float_quiet(options, "show_receptive_field", 0);

  n = n->next;
  int count = 0;
  free_section(s);

  // find l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
  node *n_tmp = n;
  int count_tmp = 0;
  if (params.train == 1)
  {
    while (n_tmp)
    {
      s = (section *)n_tmp->val;
      options = s->options;
      int stopbackward = option_find_int_quiet(options, "stopbackward", 0);
      if (stopbackward == 1)
      {
        last_stop_backward = count_tmp;
        printf("last_stop_backward = %d \n", last_stop_backward);
      }
      n_tmp = n_tmp->next;
      ++count_tmp;
    }
  }

  int old_params_train = params.train;
  int fused_layers = 0;
  fprintf(
      stderr,
      "   layer   filters  size/strd(dil)      input                output\n");

  int start_layer_index = partition_param.from;

  int end_layer_index = partition_param.to;

  int layer_index = 0;
  while (n)
  {
    if (layer_index < start_layer_index || layer_index > end_layer_index)
    {
      s = (section *)n->val;
      free_section(s);
      n = n->next;
      layer_index++;
      continue;
    }
    layer_index++;
    params.train = old_params_train;
    if (count < last_stop_backward)
      params.train = 0;

    params.index = count;
    fprintf(stderr, "%4d ", count);
    s = (section *)n->val;
    options = s->options;
    layer l = {(LAYER_TYPE)0};
    LAYER_TYPE lt = string_to_layer_type(s->type);
    if (lt == CONVOLUTIONAL)
    {
      l = parse_convolutional(options, params);
    }
    else if (lt == LOCAL)
    {
      l = parse_local(options, params);
    }
    else if (lt == ACTIVE)
    {
      l = parse_activation(options, params);
    }
    else if (lt == RNN)
    {
      l = parse_rnn(options, params);
    }
    else if (lt == GRU)
    {
      l = parse_gru(options, params);
    }
    else if (lt == LSTM)
    {
      l = parse_lstm(options, params);
    }
    else if (lt == CONV_LSTM)
    {
      l = parse_conv_lstm(options, params);
    }
    else if (lt == HISTORY)
    {
      l = parse_history(options, params);
    }
    else if (lt == CRNN)
    {
      l = parse_crnn(options, params);
    }
    else if (lt == CONNECTED)
    {
      l = parse_connected(options, params);
    }
    else if (lt == CROP)
    {
      l = parse_crop(options, params);
    }
    else if (lt == COST)
    {
      l = parse_cost(options, params);
      l.keep_delta_gpu = 1;
    }
    else if (lt == REGION)
    {
      l = parse_region(options, params);
      l.keep_delta_gpu = 1;
    }
    else if (lt == YOLO)
    {
      l = parse_yolo(options, params);
      l.keep_delta_gpu = 1;
    }
    else if (lt == GAUSSIAN_YOLO)
    {
      l = parse_gaussian_yolo(options, params);
      l.keep_delta_gpu = 1;
    }
    else if (lt == DETECTION)
    {
      l = parse_detection(options, params);
    }
    else if (lt == SOFTMAX)
    {
      l = parse_softmax(options, params);
      net.hierarchy = l.softmax_tree;
      l.keep_delta_gpu = 1;
    }
    else if (lt == CONTRASTIVE)
    {
      l = parse_contrastive(options, params);
      l.keep_delta_gpu = 1;
    }
    else if (lt == NORMALIZATION)
    {
      l = parse_normalization(options, params);
    }
    else if (lt == BATCHNORM)
    {
      l = parse_batchnorm(options, params);
    }
    else if (lt == MAXPOOL)
    {
      l = parse_maxpool(options, params);
    }
    else if (lt == LOCAL_AVGPOOL)
    {
      l = parse_local_avgpool(options, params);
    }
    else if (lt == REORG)
    {
      l = parse_reorg(options, params);
    }
    else if (lt == REORG_OLD)
    {
      l = parse_reorg_old(options, params);
    }
    else if (lt == AVGPOOL)
    {
      l = parse_avgpool(options, params);
    }
    else if (lt == ROUTE)
    {
      l = parse_route(options, params);
      int k;
      for (k = 0; k < l.n; ++k)
      {
        net.layers[l.input_layers[k]].use_bin_output = 0;
        if (count >= last_stop_backward)
          net.layers[l.input_layers[k]].keep_delta_gpu = 1;
      }
    }
    else if (lt == UPSAMPLE)
    {
      l = parse_upsample(options, params, net);
    }
    else if (lt == SHORTCUT)
    {
      l = parse_shortcut(options, params, net);
      net.layers[count - 1].use_bin_output = 0;
      net.layers[l.index].use_bin_output = 0;
      if (count >= last_stop_backward)
        net.layers[l.index].keep_delta_gpu = 1;
    }
    else if (lt == SCALE_CHANNELS)
    {
      l = parse_scale_channels(options, params, net);
      net.layers[count - 1].use_bin_output = 0;
      net.layers[l.index].use_bin_output = 0;
      net.layers[l.index].keep_delta_gpu = 1;
    }
    else if (lt == SAM)
    {
      l = parse_sam(options, params, net);
      net.layers[count - 1].use_bin_output = 0;
      net.layers[l.index].use_bin_output = 0;
      net.layers[l.index].keep_delta_gpu = 1;
    }
    else if (lt == IMPLICIT)
    {
      l = parse_implicit(options, params, net);
    }
    else if (lt == DROPOUT)
    {
      l = parse_dropout(options, params);
      l.output = net.layers[count - 1].output;
      l.delta = net.layers[count - 1].delta;
#ifdef GPU
      l.output_gpu = net.layers[count - 1].output_gpu;
      l.delta_gpu = net.layers[count - 1].delta_gpu;
      l.keep_delta_gpu = 1;
#endif
    }
    else if (lt == EMPTY)
    {
      layer empty_layer = {(LAYER_TYPE)0};
      l = empty_layer;
      l.type = EMPTY;
      l.w = l.out_w = params.w;
      l.h = l.out_h = params.h;
      l.c = l.out_c = params.c;
      l.batch = params.batch;
      l.inputs = l.outputs = params.inputs;
      l.output = net.layers[count - 1].output;
      l.delta = net.layers[count - 1].delta;
      l.forward = empty_func;
      l.backward = empty_func;
#ifdef GPU
      l.output_gpu = net.layers[count - 1].output_gpu;
      l.delta_gpu = net.layers[count - 1].delta_gpu;
      l.keep_delta_gpu = 1;
      l.forward_gpu = empty_func;
      l.backward_gpu = empty_func;
#endif
      fprintf(stderr, "empty \n");
    }
    else
    {
      fprintf(stderr, "Type not recognized: %s\n", s->type);
    }

    // calculate receptive field
    if (show_receptive_field)
    {
      int dilation = max_val_cmp(1, l.dilation);
      int stride = max_val_cmp(1, l.stride);
      int size = max_val_cmp(1, l.size);

      if (l.type == UPSAMPLE || (l.type == REORG))
      {

        l.receptive_w = receptive_w;
        l.receptive_h = receptive_h;
        l.receptive_w_scale = receptive_w_scale = receptive_w_scale / stride;
        l.receptive_h_scale = receptive_h_scale = receptive_h_scale / stride;
      }
      else
      {
        if (l.type == ROUTE)
        {
          receptive_w = receptive_h = receptive_w_scale = receptive_h_scale = 0;
          int k;
          for (k = 0; k < l.n; ++k)
          {
            layer route_l = net.layers[l.input_layers[k]];
            receptive_w = max_val_cmp(receptive_w, route_l.receptive_w);
            receptive_h = max_val_cmp(receptive_h, route_l.receptive_h);
            receptive_w_scale =
                max_val_cmp(receptive_w_scale, route_l.receptive_w_scale);
            receptive_h_scale =
                max_val_cmp(receptive_h_scale, route_l.receptive_h_scale);
          }
        }
        else
        {
          int increase_receptive = size + (dilation - 1) * 2 - 1; // stride;
          increase_receptive = max_val_cmp(0, increase_receptive);

          receptive_w += increase_receptive * receptive_w_scale;
          receptive_h += increase_receptive * receptive_h_scale;
          receptive_w_scale *= stride;
          receptive_h_scale *= stride;
        }

        l.receptive_w = receptive_w;
        l.receptive_h = receptive_h;
        l.receptive_w_scale = receptive_w_scale;
        l.receptive_h_scale = receptive_h_scale;
      }
      // printf(" size = %d, dilation = %d, stride = %d, receptive_w = %d,
      // receptive_w_scale = %d - ", size, dilation, stride, receptive_w,
      // receptive_w_scale);

      int cur_receptive_w = receptive_w;
      int cur_receptive_h = receptive_h;

      fprintf(stderr, "%4d - receptive field: %d x %d \n", count,
              cur_receptive_w, cur_receptive_h);
    }

#ifdef GPU
    // futher GPU-memory optimization: net.optimized_memory == 2
    l.optimized_memory = net.optimized_memory;
    if (net.optimized_memory == 1 && params.train && l.type != DROPOUT)
    {
      if (l.delta_gpu)
      {
        cuda_free(l.delta_gpu);
        l.delta_gpu = NULL;
      }
    }
    else if (net.optimized_memory >= 2 && params.train && l.type != DROPOUT)
    {
      if (l.output_gpu)
      {
        cuda_free(l.output_gpu);
        // l.output_gpu = cuda_make_array_pinned(l.output, l.batch*l.outputs);
        // // l.steps
        l.output_gpu = cuda_make_array_pinned_preallocated(
            NULL, l.batch * l.outputs); // l.steps
      }
      if (l.activation_input_gpu)
      {
        cuda_free(l.activation_input_gpu);
        l.activation_input_gpu = cuda_make_array_pinned_preallocated(
            NULL, l.batch * l.outputs); // l.steps
      }

      if (l.x_gpu)
      {
        cuda_free(l.x_gpu);
        l.x_gpu = cuda_make_array_pinned_preallocated(
            NULL, l.batch * l.outputs); // l.steps
      }

      // maximum optimization
      if (net.optimized_memory >= 3 && l.type != DROPOUT)
      {
        if (l.delta_gpu)
        {
          cuda_free(l.delta_gpu);
          // l.delta_gpu = cuda_make_array_pinned_preallocated(NULL,
          // l.batch*l.outputs); // l.steps printf("\n\n PINNED DELTA GPU = %d
          // \n", l.batch*l.outputs);
        }
      }

      if (l.type == CONVOLUTIONAL)
      {
        set_specified_workspace_limit(
            &l, net.workspace_size_limit); // workspace size limit 1 GB
      }
    }
#endif // GPU

    l.clip = option_find_float_quiet(options, "clip", 0);
    l.dynamic_minibatch = net.dynamic_minibatch;
    l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
    l.dont_update = option_find_int_quiet(options, "dont_update", 0);
    l.burnin_update = option_find_int_quiet(options, "burnin_update", 0);
    l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
    l.train_only_bn = option_find_int_quiet(options, "train_only_bn", 0);
    l.dontload = option_find_int_quiet(options, "dontload", 0);
    l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
    l.learning_rate_scale =
        option_find_float_quiet(options, "learning_rate", 1);
    option_unused(options);

    if (l.stopbackward == 1)
      printf(" ------- previous layers are frozen ------- \n");

    net.layers[count] = l;
    if (l.workspace_size > workspace_size)
      workspace_size = l.workspace_size;
    if (l.inputs > max_inputs)
      max_inputs = l.inputs;
    if (l.outputs > max_outputs)
      max_outputs = l.outputs;
    free_section(s);
    n = n->next;
    ++count;
    if (n)
    {
      if (l.antialiasing)
      {
        params.h = l.input_layer->out_h;
        params.w = l.input_layer->out_w;
        params.c = l.input_layer->out_c;
        params.inputs = l.input_layer->outputs;
      }
      else
      {
        params.h = l.out_h;
        params.w = l.out_w;
        params.c = l.out_c;
        params.inputs = l.outputs;
      }
    }
    if (l.bflops > 0)
      bflops += l.bflops;

    if (l.w > 1 && l.h > 1)
    {
      avg_outputs += l.outputs;
      avg_counter++;
    }
  }

  if (last_stop_backward > -1)
  {
    int k;
    for (k = 0; k < last_stop_backward; ++k)
    {
      layer l = net.layers[k];
      if (l.keep_delta_gpu)
      {
        if (!l.delta)
        {
          net.layers[k].delta =
              (float *)xcalloc(l.outputs * l.batch, sizeof(float));
        }
#ifdef GPU
        if (!l.delta_gpu)
        {
          net.layers[k].delta_gpu =
              (float *)cuda_make_array(NULL, l.outputs * l.batch);
        }
#endif
      }

      net.layers[k].onlyforward = 1;
      net.layers[k].train = 0;
    }
  }

  free_list(sections);

#ifdef GPU
  if (net.optimized_memory && params.train)
  {
    int k;
    for (k = 0; k < net.n; ++k)
    {
      layer l = net.layers[k];
      // delta GPU-memory optimization: net.optimized_memory == 1
      if (!l.keep_delta_gpu)
      {
        const size_t delta_size = l.outputs * l.batch; // l.steps
        if (net.max_delta_gpu_size < delta_size)
        {
          net.max_delta_gpu_size = delta_size;
          if (net.global_delta_gpu)
            cuda_free(net.global_delta_gpu);
          if (net.state_delta_gpu)
            cuda_free(net.state_delta_gpu);
          assert(net.max_delta_gpu_size > 0);
          net.global_delta_gpu =
              (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
          net.state_delta_gpu =
              (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
        }
        if (l.delta_gpu)
        {
          if (net.optimized_memory >= 3)
          {
          }
          else
            cuda_free(l.delta_gpu);
        }
        l.delta_gpu = net.global_delta_gpu;
      }
      else
      {
        if (!l.delta_gpu)
          l.delta_gpu = (float *)cuda_make_array(NULL, l.outputs * l.batch);
      }

      // maximum optimization
      if (net.optimized_memory >= 3 && l.type != DROPOUT)
      {
        if (l.delta_gpu && l.keep_delta_gpu)
        {
          // cuda_free(l.delta_gpu);   // already called above
          l.delta_gpu = cuda_make_array_pinned_preallocated(
              NULL, l.batch * l.outputs); // l.steps
          // printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
        }
      }

      net.layers[k] = l;
    }
  }
#endif

  set_train_only_bn(net); // set l.train_only_bn for all required layers

  net.outputs = get_network_output_size(net);
  net.output = get_network_output(net);
  avg_outputs = avg_outputs / avg_counter;
  fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
  fprintf(stderr, "avg_outputs = %d \n", avg_outputs);
#ifdef GPU
  get_cuda_stream();
  // get_cuda_memcpy_stream();
  if (gpu_index >= 0)
  {
    int size = get_network_input_size(net) * net.batch;
    net.input_state_gpu = cuda_make_array(0, size);
    if (cudaSuccess == cudaHostAlloc(&net.input_pinned_cpu,
                                     size * sizeof(float),
                                     cudaHostRegisterMapped))
      net.input_pinned_cpu_flag = 1;
    else
    {
      cudaGetLastError(); // reset CUDA-error
      net.input_pinned_cpu = (float *)xcalloc(size, sizeof(float));
    }

    // pre-allocate memory for inference on Tensor Cores (fp16)
    *net.max_input16_size = 0;
    *net.max_output16_size = 0;
    if (net.cudnn_half)
    {
      *net.max_input16_size = max_inputs;
      CHECK_CUDA(
          cudaMalloc((void **)net.input16_gpu,
                     *net.max_input16_size * sizeof(short))); // sizeof(half)
      *net.max_output16_size = max_outputs;
      CHECK_CUDA(
          cudaMalloc((void **)net.output16_gpu,
                     *net.max_output16_size * sizeof(short))); // sizeof(half)
    }
    if (workspace_size)
    {
      fprintf(stderr, " Allocate additional workspace_size = %1.2f MB \n",
              (float)workspace_size / 1000000);
      net.workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
    }
    else
    {
      net.workspace = (float *)xcalloc(1, workspace_size);
    }
  }
#else
  if (workspace_size)
  {
    net.workspace = (float *)xcalloc(1, workspace_size);
  }
#endif

  LAYER_TYPE lt = net.layers[net.n - 1].type;
  if ((net.w % 32 != 0 || net.h % 32 != 0) &&
      (lt == YOLO || lt == REGION || lt == DETECTION))
  {
    printf("\n Warning: width=%d and height=%d in cfg-file must be divisible "
           "by 32 for default networks Yolo v1/v2/v3!!! \n\n",
           net.w, net.h);
  }
  return net;
}

void load_sub_nets_weights(network net, std::vector<std::vector<network>> &sub_nets,
                           char *cfg_file, char *weights, int stages,
                           std::vector<partition_parameter> partition_params)
{
  for (int i = 0; i < stages; ++i)
  {
    for (int j = 0; j < partition_params[i].partitions; ++j)
    {
      load_weights_upto_subnet(&net, &sub_nets[i][j], weights,
                               partition_params[i].to, partition_params[i].from,
                               partition_params[i].to);
    }
  }
}

void load_weights_upto_subnet(network *net, network *sub_net, char *filename,
                              int cutoff, int start_layer, int end_layer)
{
#ifdef GPU
  if (net->gpu_index >= 0)
  {
    cuda_set_device(net->gpu_index);
  }
  if (sub_net->gpu_index >= 0)
  {
    cuda_set_device(sub_net->gpu_index);
  }
#endif
  fprintf(stderr, "Loading weights from %s...", filename);
  fflush(stdout);
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    file_error(filename);

  int major;
  int minor;
  int revision;
  fread(&major, sizeof(int), 1, fp);
  fread(&minor, sizeof(int), 1, fp);
  fread(&revision, sizeof(int), 1, fp);
  if ((major * 10 + minor) >= 2)
  {
    printf("\n seen 64");
    uint64_t iseen = 0;
    fread(&iseen, sizeof(uint64_t), 1, fp);
    *net->seen = iseen;
    *sub_net->seen = iseen;
  }
  else
  {
    printf("\n seen 32");
    uint32_t iseen = 0;
    fread(&iseen, sizeof(uint32_t), 1, fp);
    *net->seen = iseen;
    *sub_net->seen = iseen;
  }
  *net->cur_iteration = get_current_batch(*net);
  printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n",
         (float)(*net->seen / 1000), (float)(*net->seen / 64000));
  int transpose = (major > 1000) || (minor > 1000);

  int i;
  for (i = 0; i < net->n; ++i)
  {
    layer l;
    if (i >= start_layer && i <= end_layer)
    {
      l = sub_net->layers[i - start_layer];
    }
    else
    {
      l = net->layers[i];
    }
    if (l.dontload)
      continue;
    if (l.type == CONVOLUTIONAL && l.share_layer == NULL)
    {
      load_convolutional_weights(l, fp);
    }
    if (l.type == SHORTCUT && l.nweights > 0)
    {
      load_shortcut_weights(l, fp);
    }
    if (l.type == IMPLICIT)
    {
      load_implicit_weights(l, fp);
    }
    if (l.type == CONNECTED)
    {
      load_connected_weights(l, fp, transpose);
    }
    if (l.type == BATCHNORM)
    {
      load_batchnorm_weights(l, fp);
    }
    if (l.type == CRNN)
    {
      load_convolutional_weights(*(l.input_layer), fp);
      load_convolutional_weights(*(l.self_layer), fp);
      load_convolutional_weights(*(l.output_layer), fp);
    }
    if (l.type == RNN)
    {
      load_connected_weights(*(l.input_layer), fp, transpose);
      load_connected_weights(*(l.self_layer), fp, transpose);
      load_connected_weights(*(l.output_layer), fp, transpose);
    }
    if (l.type == GRU)
    {
      load_connected_weights(*(l.input_z_layer), fp, transpose);
      load_connected_weights(*(l.input_r_layer), fp, transpose);
      load_connected_weights(*(l.input_h_layer), fp, transpose);
      load_connected_weights(*(l.state_z_layer), fp, transpose);
      load_connected_weights(*(l.state_r_layer), fp, transpose);
      load_connected_weights(*(l.state_h_layer), fp, transpose);
    }
    if (l.type == LSTM)
    {
      load_connected_weights(*(l.wf), fp, transpose);
      load_connected_weights(*(l.wi), fp, transpose);
      load_connected_weights(*(l.wg), fp, transpose);
      load_connected_weights(*(l.wo), fp, transpose);
      load_connected_weights(*(l.uf), fp, transpose);
      load_connected_weights(*(l.ui), fp, transpose);
      load_connected_weights(*(l.ug), fp, transpose);
      load_connected_weights(*(l.uo), fp, transpose);
    }
    if (l.type == CONV_LSTM)
    {
      if (l.peephole)
      {
        load_convolutional_weights(*(l.vf), fp);
        load_convolutional_weights(*(l.vi), fp);
        load_convolutional_weights(*(l.vo), fp);
      }
      load_convolutional_weights(*(l.wf), fp);
      if (!l.bottleneck)
      {
        load_convolutional_weights(*(l.wi), fp);
        load_convolutional_weights(*(l.wg), fp);
        load_convolutional_weights(*(l.wo), fp);
      }
      load_convolutional_weights(*(l.uf), fp);
      load_convolutional_weights(*(l.ui), fp);
      load_convolutional_weights(*(l.ug), fp);
      load_convolutional_weights(*(l.uo), fp);
    }
    if (l.type == LOCAL)
    {
      int locations = l.out_w * l.out_h;
      int size = l.size * l.size * l.c * l.n * locations;
      fread(l.biases, sizeof(float), l.outputs, fp);
      fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
      if (gpu_index >= 0)
      {
        push_local_layer(l);
      }
#endif
    }
    if (feof(fp))
      break;
  }
  fprintf(stderr, "Done! Loaded %d layers from weights-file \n", i);
  fclose(fp);
}