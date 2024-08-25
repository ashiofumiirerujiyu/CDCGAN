import yaml
import torch
from model import Generator


def export_CDCGAN_to_onnx(model, noise, label, onnx_path="model.onnx"):
    torch.onnx.export(
        model,
        (noise, label),
        onnx_path,
        input_names=["noise", "label"],
        output_names=["output"],
        dynamic_axes={
            "noise": {0: "batch_size"},  # 배치 크기를 유동적으로 설정
            "label": {0: "batch_size"},  # 배치 크기를 유동적으로 설정
            "output": {0: "batch_size"}  # 배치 크기를 유동적으로 설정
        }
    )
    print(f"CDCGAN Generator has been successfully exported to {onnx_path}")

yaml_path = "./config/onnx.yaml"
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

model = Generator(config['model']['generator']['noise_dim'], config['model']['generator']['label_dim'], config['model']['generator']['img_channels'], config['model']['generator']['feature_g'], config['model']['generator']['dropout_p'])
model.load_state_dict(torch.load(config['model']['generator']['weight_path'], weights_only=True))

noise = torch.randn(1, 100)
label = torch.tensor(1).unsqueeze(0)

export_CDCGAN_to_onnx(model, noise, label)