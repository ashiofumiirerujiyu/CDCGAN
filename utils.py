import os
import pytz
import random
import logging
import torch
import onnx
from datetime import datetime


class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        kst = pytz.timezone('Asia/Seoul')
        log_time = datetime.fromtimestamp(record.created, kst)
        if datefmt:
            return log_time.strftime(datefmt)
        else:
            return log_time.strftime('%Y-%m-%d %H:%M:%S')
    

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_output_path(seed):
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    base_output_path = os.path.join(script_dir, "output")
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path, exist_ok=True)

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)

    date_str = now.strftime("%Y_%m_%d")
    time_str = now.strftime("%H%M%S")

    output_path = os.path.join(base_output_path, f"{date_str}", f"{time_str}_{seed}")
    os.makedirs(output_path, exist_ok=True)

    return output_path


def load_and_check_onnx_model(model_path):
    # ONNX 모델 로드
    onnx_model = onnx.load(model_path)
    
    # 모델 체크
    onnx.checker.check_model(onnx_model)
    
    # 모델 입력 및 출력 차원 출력
    print("Model inputs:")
    for input in onnx_model.graph.input:
        input_dims = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        print(f"{input.name}: {input_dims}")
    
    print("Model outputs:")
    for output in onnx_model.graph.output:
        output_dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"{output.name}: {output_dims}")
        