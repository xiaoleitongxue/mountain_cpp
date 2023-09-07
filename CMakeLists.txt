cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(mountain_cpp)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

option(ENABLE_CUDA "Enable CUDA support" OFF)

# find package with config mode
set(Torch_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED CONFIGS TorchConfig.cmake)

set(Darknet_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/share/darknet)
find_package(Darknet REQUIRED)

set(nlohmann_json_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/json/install/share/cmake/nlohmann_json)
find_package(nlohmann_json CONFIGS REQUIRED nlohmann_jsonConfig.cmake)


add_library(data_packet src/data_packet.cpp)
target_include_directories(data_packet PRIVATE src)
target_link_libraries(data_packet PRIVATE "${TORCH_LIBRARIES}")

add_library(partition_model src/partition_model.cpp)
target_include_directories(partition_model
        PUBLIC src ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src)
target_link_libraries(
        partition_model PUBLIC Darknet::dark)
if (ENABLE_CUDA)
    target_link_libraries(
            partition_model PUBLIC -L/usr/local/cuda/lib64 -lcuda -lcudart
            -lcublas -lcurand)
endif ()

add_library(parse_launch_config src/parse_launch_config.cpp src/parse_launch_config.hpp)
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
if(ENABLE_CUDA)
    target_compile_definitions(worker PUBLIC GPU)
endif()
target_include_directories(
        worker PRIVATE ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src
        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb/include src)
target_link_libraries(worker PRIVATE data_packet "${TORCH_LIBRARIES}"
        partition_model Darknet::dark)

add_executable(main src/main.cpp)
if(ENABLE_CUDA)
    target_compile_definitions(main PUBLIC GPU)
endif()
target_include_directories(
        main PUBLIC ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src
        ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb/include src)
#target_link_options(main PRIVATE -Wl,-rpath,../lib)
target_link_libraries(
        main PUBLIC worker parse_launch_config nlohmann_json::nlohmann_json partition_model inference_helper
         Darknet::dark)


# install executable
# set(CMAKE_INSTALL_RPATH ${PROJECT_SOURCE_DIR}/3rdparty/lib)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/mc_install)


## install darknet shared library
#install(IMPORTED_RUNTIME_ARTIFACTS Darknet::dark DESTINATION lib)

# # install libtorch shared library
# install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libtorch/lib DESTINATION ./)

## set where main to find shared library
#SET_TARGET_PROPERTIES(main PROPERTIES INSTALL_RPATH "$ORIGIN;../lib")
# install executable
install(TARGETS main RUNTIME DESTINATION bin)

## install image
#install(FILES ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/data/dog.jpg DESTINATION data)


## install config
#install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/config DESTINATION ./)

## install weights
#install(FILES ${CMAKE_CURRENT_LIST_DIR}/data/yolov2-tiny.weights DESTINATION data)
