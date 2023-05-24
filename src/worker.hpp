#ifndef WORKER_HPP
#define WORKER_HPP
#include <darknet.h>
#include <queue>
#include <string>
#include <vector>
#include <data_packet.hpp>
class worker {
private:
  std::string ip;
  int port;
  std::vector<network> nets;
  std::priority_queue<data_packet> p_queue;
public:
 worker(std::string ip_, int port, std::vector<network> nets);
 void inference();
 void sent_data_packet();
 void receive_data_packet();
};
#endif