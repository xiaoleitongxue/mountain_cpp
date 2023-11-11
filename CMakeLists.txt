cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(mountain_cpp)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_GENERATOR Ninja)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
option(ENABLE_CUDA "Enable CUDA support" OFF)

# find package with config mode
set(Torch_DIR "C:/Users/postb/miniconda3/Lib/site-packages/torch/share/cmake")
find_package(Torch REQUIRED PATHS "C:/Users/lilei/miniconda3/Lib/site-packages/torch/share/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# # For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)



# 在外部项目构建后，添加一个自定义构建步骤
ExternalProject_Add(
        darknet
        URL https://github.com/AlexeyAB/darknet/archive/refs/tags/yolov4.zip
        CMAKE_ARGS
        "-DCMAKE_BUILD_TYPE=Release"
        "-DENABLE_OPENCV=OFF"
        "-DENABLE_CUDA=OFF"
        DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps/darknet-download"
        SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/darknet-src"
        BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/darknet-build"
        INSTALL_DIR "${CMAKE_BINARY_DIR}/_deps/darknet-src"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        BUILD_ALWAYS TRUE
)

ExternalProject_Get_Property(darknet INSTALL_DIR)
set(DARKNET_INSTALL_DIR ${INSTALL_DIR})
message("Install dir of darknet = ${DARKNET_INSTALL_DIR}")

set(DARKNET_SRC_INCLUDE_DIR ${DARKNET_INSTALL_DIR}/src)
set(DARKNET_INCLUDE_DIR ${DARKNET_INSTALL_DIR}/include)

set(CMAKE_MODULE_PATH "${DARKNET_INSTALL_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

set(Stb_DIR ${DARKNET_INSTALL_DIR}/3rdparty/stb CACHE PATH "Path where Stb image library can be located")

find_package(Threads REQUIRED)
set(DARKNET_LIB_DIR "${DARKNET_INSTALL_DIR}")
set(prefix "lib")
set(suffix ".a")

set(DARKNET_LIBRARIES
        "${DARKNET_LIB_DIR}/${prefix}darknet${suffix}"
)

# set(Darknet_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/share/darknet)
# find_package(Darknet REQUIRED)
# include(FetchContent)


add_library(data_packet src/data_packet.cpp)
target_include_directories(data_packet PRIVATE src)
target_link_libraries(data_packet PRIVATE "${TORCH_LIBRARIES}")

add_library(partition_model src/partition_model.cpp)
target_include_directories(partition_model
        PUBLIC src ${DARKNET_SRC_INCLUDE_DIR} ${DARKNET_INCLUDE_DIR})
target_compile_definitions(partition_model PRIVATE -D_TIMESPEC_DEFINED)
target_link_libraries(
        partition_model PUBLIC ${DARKNET_LIBRARIES})

if(ENABLE_CUDA)
        target_link_libraries(
                partition_model PUBLIC -L/usr/local/cuda/lib64 -lcuda -lcudart
                -lcublas -lcurand)
endif()

add_library(parse_launch_config src/parse_launch_config.cpp src/parse_launch_config.hpp)
target_include_directories(parse_launch_config
        PUBLIC src ${DARKNET_SRC_INCLUDE_DIR} ${DARKNET_INCLUDE_DIR})
target_compile_definitions(parse_launch_config PRIVATE -D_TIMESPEC_DEFINED)
target_link_libraries(
        parse_launch_config PUBLIC partition_model nlohmann_json::nlohmann_json
        ${DARKNET_LIBRARIES})

add_library(inference_helper src/inference_helper.cpp)
target_include_directories(inference_helper
        PUBLIC src ${DARKNET_SRC_INCLUDE_DIR} ${DARKNET_INCLUDE_DIR})
target_compile_definitions(inference_helper PRIVATE -D_TIMESPEC_DEFINED)
target_link_libraries(inference_helper PUBLIC partition_model ${DARKNET_LIBRARIES})

add_library(worker src/worker.cpp)

if(ENABLE_CUDA)
        target_compile_definitions(worker PUBLIC GPU)
endif()

message(${Stb_DIR})
target_include_directories(
        worker PRIVATE ${DARKNET_SRC_INCLUDE_DIR} ${DARKNET_INCLUDE_DIR}
        ${Stb_DIR}/include src)
target_compile_definitions(worker PRIVATE -D_TIMESPEC_DEFINED)
target_link_libraries(worker PRIVATE data_packet "${TORCH_LIBRARIES}"
        partition_model ${DARKNET_LIBRARIES})

# add_executable(main src/main.cpp)

# if(ENABLE_CUDA)
# target_compile_definitions(main PUBLIC GPU)
# endif()

# target_include_directories(
# main PUBLIC ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/src
# ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb/include src)

# # target_link_options(main PRIVATE -Wl,-rpath,../lib)
# target_link_libraries(
# main PUBLIC worker parse_launch_config nlohmann_json::nlohmann_json partition_model inference_helper
# Darknet::dark)

# # install executable
# # set(CMAKE_INSTALL_RPATH ${PROJECT_SOURCE_DIR}/3rdparty/lib)
# set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/mc_install)

# # # install darknet shared library
# # install(IMPORTED_RUNTIME_ARTIFACTS Darknet::dark DESTINATION lib)

# # # install libtorch shared library
# # install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/3rdparty/libtorch/lib/ DESTINATION ./)

# # # set where main to find shared library
# # SET_TARGET_PROPERTIES(main PROPERTIES INSTALL_RPATH "$ORIGIN;../lib")
# # install executable
# install(TARGETS main RUNTIME DESTINATION bin)

# # ## install image
# # install(FILES ${CMAKE_CURRENT_LIST_DIR}/3rdparty/darknet/data/dog.jpg DESTINATION data)

# # ## install config
# # install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/config DESTINATION ./)

# # # install weights
# # install(FILES ${CMAKE_CURRENT_LIST_DIR}/data/yolov2-tiny.weights DESTINATION data)
