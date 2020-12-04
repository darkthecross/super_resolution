cmake_minimum_required(VERSION 3.5)

project(super_resolution)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

#####################################################################################

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Add protos.
ADD_SUBDIRECTORY(src/proto)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

add_executable(generate_training_examples src/generate_training_examples.cc)
target_link_libraries(generate_training_examples training_example_proto ${OpenCV_LIBS} )
