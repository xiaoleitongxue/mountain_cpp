cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mountain_cpp)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(Torch_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED CONFIGS TorchConfig.cmake)

# set(CMAKE_C_COMPILER clang-16) set(CMAKE_CXX_COMPILER clang++-16)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++
# -lc++abi") set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_compile_options(-fcolor-diagnostics)

set(Darknet_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/share/darknet)
find_package(Darknet REQUIRED)

set(nlohmann_json_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/json/build)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(data_packet src/data_packet.cpp)
target_include_directories(data_packet PRIVATE src)
target_link_libraries(data_packet PRIVATE "${TORCH_LIBRARIES}")

add_library(partition_model src/partition_model.cpp)
target_include_directories(partition_model
                           PUBLIC src ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src)
target_link_libraries(
  partition_model PUBLIC Darknet::dark -L/usr/local/cuda/lib64 -lcuda -lcudart
                         -lcublas -lcurand)

add_library(parse_launch_config src/parse_launch_config.cpp)
target_include_directories(parse_launch_config
                           PUBLIC src ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src)
target_link_libraries(
  parse_launch_config PUBLIC partition_model nlohmann_json::nlohmann_json
                             Darknet::dark)

add_library(inference_helper src/inference_helper.cpp)
target_include_directories(inference_helper
                           PUBLIC src ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src)

target_link_libraries(inference_helper PUBLIC partition_model Darknet::dark)

add_library(worker src/worker.cpp)
target_compile_definitions(worker PUBLIC GPU)
target_include_directories(
  worker PRIVATE ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src
                 ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb/include src)
target_link_libraries(worker PRIVATE data_packet "${TORCH_LIBRARIES}"
                                     partition_model Darknet::dark)

# set_property(TARGET worker PROPERTY CXX_STANDARD 14)

add_executable(main src/main.cpp)
target_compile_definitions(worker PUBLIC GPU)
target_include_directories(
  main PUBLIC ${CMAKE_CURRENT_LIST_DIR}//3rdparty/darknet/src
              ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb/include src)
target_link_libraries(
  main PUBLIC worker partition_model parse_launch_config inference_helper
              nlohmann_json::nlohmann_json Darknet::dark)
              set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/launch)
install(TARGETS main DESTINATION bin)
