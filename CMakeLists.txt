cmake_minimum_required(VERSION 3.22)
project(mountain_cpp)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(Darknet_DIR ${CMAKE_CURRENT_LIST_DIR}/darknet/share/darknet)
find_package(Darknet REQUIRED)
set(nlohmann_json_DIR ${CMAKE_CURRENT_LIST_DIR}/json/build)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH /home/wsl/src/libtorch)
find_package(Torch REQUIRED)

add_library(partition_model src/partition_model.cpp)
target_compile_definitions(partition_model PUBLIC GPU)
target_include_directories(partition_model PUBLIC src ${CMAKE_CURRENT_LIST_DIR}/darknet/src)
target_link_libraries(partition_model PUBLIC Darknet::dark -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand)


add_library(parse_launch_config src/parse_launch_config.cpp)
target_include_directories(parse_launch_config PUBLIC src partition_model)
target_link_libraries(parse_launch_config PUBLIC partition_model nlohmann_json::nlohmann_json)

add_executable(parse_json_test src/parse_json_test.cpp)
target_include_directories(parse_json_test PUBLIC src partition_model)
target_link_libraries(parse_json_test PRIVATE parse_launch_config partition_model)

add_library(inference_helper src/inference_helper.cpp)
target_include_directories(inference_helper PUBLIC src partition_model ${OpenCV_INCLUDE_DIRS})
target_link_libraries(inference_helper PUBLIC partition_model Darknet::dark ${OpenCV_LIBS})



add_library(worker src/worker.cpp)
target_include_directories(worker PUBLIC src ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}/darknet/src ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb/include)
target_link_libraries(worker Darknet::dark ${OpenCV_LIBS} partition_model "${TORCH_LIBRARIES}")

add_executable(main src/main.cpp)
target_compile_definitions(main PUBLIC GPU)
target_include_directories(main PRIVATE src)
target_link_libraries(main PUBLIC parse_launch_config inference_helper partition_model)