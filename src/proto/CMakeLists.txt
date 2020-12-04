INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)

INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER training_example.proto)
ADD_LIBRARY(training_example_proto ${PROTO_HEADER} ${PROTO_SRC})
target_link_libraries(training_example_proto ${Protobuf_LIBRARIES})