# To run the following command, grpcio-tools is needed.
# if it's not installed, run "pip install grpcio-tools"
python -m grpc_tools.protoc -I=./ --python_out=./ --grpc_python_out=./ ./management.proto
