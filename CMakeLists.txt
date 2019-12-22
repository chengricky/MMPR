cmake_minimum_required(VERSION 3.0.0)

project(SeqPR VERSION 0.1.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(CMAKE_BUILD_TYPE "Debug")

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SeqPR_SOURCE_DIR}/bin)



find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIR_SCRS)
add_subdirectory(Descriptors)
add_subdirectory(Tools)

add_executable(SeqPR ${DIR_SCRS})

# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(SeqPR ${OpenCV_LIBS} tools descs)

