#ifndef DATA_BLOB_HPP
#define DATA_BLOB_HPP
typedef struct data_packet {
  int data_size; 
  int frame_seq;
  int task_id;
  int stage;
  int from;
  int to;
  int w;
  int h;
  int c;
  float *data;
}data_packet;
#endif