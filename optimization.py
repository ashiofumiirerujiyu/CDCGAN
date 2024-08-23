import time
import torch
import tensorrt as trt
from model import Generator

# 모델 준비
model = Generator(100, 10, 784)
model.load_state_dict(torch.load("/workspace/cGAN/output/2024_08_23/132557_42/best_generator.pth"))
model.eval()

# 입력 데이터 준비
noise = torch.randn(1, 100)
label = torch.tensor(1)
label = label.unsqueeze(0)

# 추론 시간 측정
start_time = time.time()
with torch.no_grad():  # Gradient 계산 비활성화
    output = model(noise, label)
end_time = time.time()

print(f"Inference Time: {end_time - start_time:.6f} seconds")




# 모델 인스턴스화
model = Generator(100, 10, 784)
model.load_state_dict(torch.load("/workspace/cGAN/output/2024_08_23/132557_42/best_generator.pth"))
model.eval()

# Dummy input for ONNX export
dummy_input = torch.randn(1, 100)
label_input = torch.tensor(1).unsqueeze(0)

# Trace model
torch.onnx.export(
    model,
    (dummy_input, label_input),
    "/workspace/cGAN/output/2024_08_23/best_generator.onnx",
    verbose=True,
    input_names=['noise', 'label'],
    output_names=['output'],
    dynamic_axes={'noise': {0: 'batch_size'}, 'label': {0: 'batch_size'}}
)



# ONNX 모델 로드
onnx_model_path = "/workspace/cGAN/output/2024_08_23/best_generator.onnx"

# TensorRT logger 및 builder 생성
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)

# ONNX 모델을 TensorRT 네트워크로 변환
with open(onnx_model_path, 'rb') as f:
    onnx_model = f.read()
parser.parse(onnx_model)

# Builder 설정
builder.max_batch_size = 1
builder.max_workspace_size = 1 << 30  # 1GB

# TensorRT 엔진 생성
engine = builder.build_cuda_engine(network)

# 실행 컨텍스트 생성
context = engine.create_execution_context()

# 추론을 위한 입력/출력 버퍼 설정
import numpy as np

# Example input data
input_data = np.random.randn(1, 100).astype(np.float32)
label_data = np.array([1]).astype(np.int64)

# Allocate device buffers
import pycuda.driver as cuda
import pycuda.autoinit

d_input = cuda.mem_alloc(input_data.nbytes)
d_label = cuda.mem_alloc(label_data.nbytes)
d_output = cuda.mem_alloc(input_data.nbytes)  # Output size needs to be adjusted

# Transfer data to GPU
cuda.memcpy_htod(d_input, input_data)
cuda.memcpy_htod(d_label, label_data)

# Perform inference
bindings = [int(d_input), int(d_label), int(d_output)]
context.execute(batch_size=1, bindings=bindings)

# Retrieve output
output_data = np.empty_like(input_data)
cuda.memcpy_dtoh(output_data, d_output)

start_time = time.time()
print("Output:", output_data)
end_time = time.time()

print(f"Inference Time: {end_time - start_time:.6f} seconds")
