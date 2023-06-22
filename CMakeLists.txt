cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(mountain_cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)


# find package with config mode
set(Torch_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libtorch/torch/share/cmake/Torch)
find_package(Torch REQUIRED CONFIGS TorchConfig.cmake)

set(Darknet_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/share/darknet)
find_package(Darknet REQUIRED)

set(nlohmann_json_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/json/build)
find_package(nlohmann_json 3.2.0 REQUIRED)



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

add_executable(main src/main.cpp)
target_compile_definitions(worker PUBLIC GPU)
target_include_directories(
  main PUBLIC ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src
              ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb/include src)
target_link_options(main PRIVATE -Wl,-rpath,../lib)
target_link_libraries(
  main PUBLIC worker partition_model parse_launch_config inference_helper
              nlohmann_json::nlohmann_json Darknet::dark)


# install executable
# set(CMAKE_INSTALL_RPATH ${PROJECT_SOURCE_DIR}/3rdparty/lib)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/install)
# install(TARGETS main DESTINATION bin)
# # install(FILES  DESTINATION lib)
# # SET_TARGET_PROPERTIES(main
# #     PROPERTIES INSTALL_RPATH "$ORIGIN;../lib")
# # install(FILES ${CMAKE_CURRENT_LIST_DIR}/build/main DESTINATION bin)
# # install(FILES Darknet::dark DESTINATION lib)

# install darknet shared library
install(IMPORTED_RUNTIME_ARTIFACTS Darknet::dark DESTINATION lib)

# install libtorch shared library
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libtorch/lib DESTINATION ./)

# set where main to find shared library
SET_TARGET_PROPERTIES(main PROPERTIES INSTALL_RPATH "$ORIGIN;../lib")
# install executable
install(TARGETS main RUNTIME DESTINATION bin)

# install image
install(FILES ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/data/dog.jpg DESTINATION data)


# install config
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/config DESTINATION ./)

# install weights
# install(FILES ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/data/dog.jpg DESTINATION data)
