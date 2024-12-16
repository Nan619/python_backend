<!--
# Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Preprocessing Using Python Backend Example
This example shows how to preprocess your inputs using Python backend before it is passed to the TensorRT model for inference. This ensemble model includes an image preprocessing model (preprocess) and a TensorRT model (resnet50_trt) to do inference.

**1. Converting PyTorch Model to ONNX format:**

Run onnx_exporter.py to convert ResNet50 PyTorch model to ONNX format. Width and height dims are fixed at 224 but dynamic axes arguments for dynamic batching are used. Commands from the 2. and 3. subsections shall be executed within this Docker container.

    docker run -it --gpus=all -v $(pwd):/workspace -v /root/.cache/torch:/root/.cache/torch nvcr.io/nvidia/pytorch:24.05-py3 bash
    pip install numpy pillow torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
    python onnx_exporter.py --save model.onnx

**2. Create the model repository:**

    mkdir -p model_repository/ensemble_python_resnet50/1
    mkdir -p model_repository/resnet50_onnx/1

    # Copy the ONNX model
    cp model.onnx model_repository/resnet50_onnx/1

**3. Run the command below to start the server container:**

Under python_backend/examples/processing, run this command to start the server docker container:

    docker run --gpus=all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:24.05-py3 bash
    pip install numpy pillow torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
    tritonserver --model-repository=/models

**5. Start the client to test:**

Under python_backend/examples/processing, run the commands below to start the client Docker container:

    docker run --rm --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:24.05-py3-sdk python client.py --image mug.jpg
    The result of classification is:COFFEE MUG

Here, since we input an image of "mug" and the inference result is "COFFEE MUG" which is correct.
