cmake_minimum_required(VERSION 3.0)

#项目名称
project(SeqPlaceRecognition)
find_package(OpenCV REQUIRED)

#头文件路径
include_directories(
    ./
)

#代码路径
aux_source_directory(. DIR_SRCS)

#子文件夹
add_subdirectory(Descriptors)
add_subdirectory(FileInterface)
add_subdirectory(Tools)

#dubug 模式
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")

#生成可执行的文件
add_executable(SeqPR ${DIR_SRCS})

#加入动态链接库
target_link_libraries(SeqPR ${OpenCV_LIBS})
