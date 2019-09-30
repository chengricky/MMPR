aux_source_directory(. DIR_DESC_SRCS)
add_library(descs ${DIR_DESC_SRCS} ../Tools/GNSSDistance.cpp ../Tools/timer.cpp)
target_link_libraries(descs z)

