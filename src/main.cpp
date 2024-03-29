#include <worker.hpp>
#include "inference_helper.hpp"

#include "parse_launch_config.hpp"
#include <string>
#include <thread>
#ifdef WIN32
#include <direct.h> // for _getcwd, _chdir, etc.
#include <io.h>     // for _write, _read, _access, etc.
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h> // for write, read, access, etc.
#endif
#include <vector>

using namespace std::chrono;
int main(int argc, char *argv[]) {

    std::string launch_json = argv[1];

    // prase json
    launch_parameter launch_param = read_init_config(launch_json);

    char *cfgfile = const_cast<char *>(launch_param.cfg.c_str());
    char *weights_file = const_cast<char *>(launch_param.weights.c_str());
    // load_network
    network* net = load_network(cfgfile, weights_file, 0);

    for (int i = 0; i < launch_param.stages; ++i) {
        int from = launch_param.partition_params[i].from;
        int to = launch_param.partition_params[i].to;
        launch_param.partition_params[i].in_w = net->layers[from].w;
        launch_param.partition_params[i].in_h = net->layers[from].h;
        launch_param.partition_params[i].in_c = net->layers[from].c;
        launch_param.partition_params[i].out_w = net->layers[to].out_w;
        launch_param.partition_params[i].out_h = net->layers[to].out_h;
        launch_param.partition_params[i].out_c = net->layers[to].out_c;
    }
    std::vector<ftp_parameter> ftp_params =
            perform_ftp(launch_param.partition_params, launch_param.stages, *net);

    std::string worker_type = argv[2];
    if (worker_type == "master") {
        const char *workers_ = argv[3];
        int workers = atoi(workers_);
        network last_stage_net = parse_network_cfg_custom_whc(
                cfgfile, launch_param.partition_params[launch_param.stages - 1],
                ftp_params[launch_param.stages - 1], 0, 1, 1);
        load_weights_upto_subnet(
                net, &last_stage_net, weights_file,
                launch_param.partition_params[launch_param.stages - 1].to,
                launch_param.partition_params[launch_param.stages - 1].from,
                launch_param.partition_params[launch_param.stages - 1].to);
        free_network(*net);
        Master master{launch_param.master_addr.ip,
                      launch_param.master_addr.port,
                      launch_param.stages,
                      last_stage_net,
                      launch_param.frames,
                      launch_param.partition_params,
                      ftp_params,
                      launch_param.worker_addr};

        // 每个worker对应一个连接
        std::vector<int> client_fds;
        for (int i = 0; i < workers; ++i) {
            int status, valread, client_fd;
            struct sockaddr_in serv_addr;

            if ((client_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
                printf("\n Socket creation error \n");
                return -1;
            }

            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(launch_param.worker_addr[i].port);

            // Convert IPv4 and IPv6 addresses from text to binary
            // form
            if (inet_pton(AF_INET, launch_param.worker_addr[i].ip.c_str(),
                          &serv_addr.sin_addr) <= 0) {
                printf("\nInvalid address/ Address not supported \n");
                return -1;
            }

            if ((status = connect(client_fd, (struct sockaddr *) &serv_addr,
                                  sizeof(serv_addr))) < 0) {
                printf("\nConnection Failed \n");
                return -1;
            }
            client_fds.push_back(client_fd);
        }

        // 每个阶段共有4个task， 计算每个worker分配到几个task
        std::vector<int> fenpei(workers, 4 / workers);
        if (4 % workers != 0) {
            fenpei[workers - 1]++;
        }

        // create thread pool
        std::vector<std::thread> send_data_packet_threads;

        for (int i = 0; i < launch_param.frames; ++i) {
            master.m_push_image(i, launch_param.filename);
            auto start = high_resolution_clock::now();
            for (int p = 0; p < launch_param.stages - 1; ++p) {
                // partition
                master.m_pritition_image();

                // inference partition
                for (int j = 0; j < fenpei.size(); ++j) {
                    std::thread send_data_packet_thread(&Master::m_send_data_packet,
                                                        &master, client_fds[j], fenpei[j]);
                    send_data_packet_threads.push_back(std::move(send_data_packet_thread));
                    // master.m_send_data_packet(client_fds[j], fenpei[j]);
                }

                for (std::thread &th: send_data_packet_threads) {
                    // If thread Object is Joinable then Join that thread.
                    if (th.joinable())
                        th.join();
                }
                master.m_merge_partitions();
            }

            // inference final phase
            master.m_inference();
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds >(stop - start);

// To get the value of duration use the count()
// member function on the duration object
            std::cout << "frame " << i << ": " << duration.count()  << "ms"<< std::endl;
        }
        for (int client_fd : client_fds) {
            close(client_fd);
        }

    } else if (worker_type == "worker") {
        std::vector<std::vector<network>> sub_nets(launch_param.stages - 1);
        // generate sub network
        for (int i = 0; i < launch_param.stages - 1; ++i) {
            for (int j = 0; j < launch_param.partition_params[i].partitions; ++j) {
                network sub_net = parse_network_cfg_custom_whc(
                        cfgfile, launch_param.partition_params[i], ftp_params[i], j, 1, 1);
                sub_nets[i].push_back(sub_net);
            }
        }

        // load weights
        load_sub_nets_weights(*net, sub_nets, cfgfile, weights_file,
                              launch_param.stages - 1,
                              launch_param.partition_params);
        free_network(*net);
        // flip weights
        flip_sub_nets_weights(sub_nets, launch_param.stages - 1,
                              launch_param.partition_params, ftp_params);

        // create worker object
        std::string worker_id = argv[3];
        int worker_id_ = std::stoi(worker_id);
        Worker worker{worker_id_, launch_param.worker_addr[worker_id_].ip,
                      launch_param.worker_addr[worker_id_].port, sub_nets,
                      launch_param.master_addr};
        // worker.m_receive_data_packet();
        // start worker thread
        std::thread recv_data_packet_thread(&Worker::m_recv_data_packet, &worker);

        std::thread inference_thread(&Worker::m_inference, &worker);

        inference_thread.join();
        recv_data_packet_thread.join();
    }


    return 0;
}
