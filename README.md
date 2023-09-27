# LLM-Inference
A Cisco Research project to optimize Large Language Model Inference systems. Internal project name: InfScale

## Code Structure
```
inference_pipeline.py -- the script that bears the PyTorch RPC-based implementation of LLM pipeline.
optim_inference_pipeline.py -- the script that bears the optimized implementation of LLM pipeline that combines both PyTorch RPC and low-level communication primitives.
partition_analysis.py -- the script that analyzes different partition strategy of ML models.
resnet50/ -- the directory that holds the scripts to run resnet50 inference and profiling as well as experiment results.
vgg16/ -- the directory that holds the scripts to run vgg16 inference and profiling as well as experiment results.
bert/ -- the directory that holds the scripts to run Bert inference and profiling as well as experiment results.
llama/ -- the directory that holds the scripts to run Llama inference and profiling results.
profiling/ -- the directory that holds the scripts to do profiling for ML models.
```
