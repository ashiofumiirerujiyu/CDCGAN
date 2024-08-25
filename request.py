import tritonclient.grpc as grpcclient
import numpy as np
import yaml


def CDCGAN_triton_infer(model_name, batch_size, label_dim, server_url="localhost:8001", num=0):
    # 클라이언트 생성
    triton_client = grpcclient.InferenceServerClient(url=server_url)
    
    # NumPy를 사용하여 noise와 label 배열 생성
    noise = np.random.randn(batch_size, 100).astype(np.float32)  # noise 배열 생성
    label = np.random.randint(num, label_dim, size=(batch_size,)).astype(np.int64)  # label 배열 생성
    
    # Triton의 입력 객체 생성
    inputs = []
    inputs.append(grpcclient.InferInput('noise', noise.shape, 'FP32'))
    inputs.append(grpcclient.InferInput('label', label.shape, 'INT64'))
    
    # 입력 데이터 설정
    inputs[0].set_data_from_numpy(noise)
    inputs[1].set_data_from_numpy(label)
    
    # 출력 객체 생성
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))
    
    # 모델 추론 요청
    response = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    # 결과 추출
    output_img = response.as_numpy('output')
    
    # 범위 변환: [-1, 1] 범위를 [0, 255]로 변경
    output_img = (output_img + 1) / 2 * 255
    
    # 데이터 타입을 uint8로 변환
    output_img = output_img.astype(np.uint8)
    print(f"output_img: {output_img.shape}")
    
    return output_img

yaml_path = "./config/request.yaml"
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

output_img = CDCGAN_triton_infer(config['request']['model_name'], config['request']['batch_size'], config['request']['label_dim'])
