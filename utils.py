import os
import pytz
import random
import logging
import torch
from datetime import datetime
from torchvision.utils import save_image


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


def save_generated_images(generator, noise_dim, save_path, epoch, y, device):
    y = y.unsqueeze(0).to(device)
    z = torch.randn(1, noise_dim)
    
    with torch.no_grad():
        fake_image = generator(z, y)
        fake_image = fake_image.view(1, 1, 28, 28)  # Adjust shape as necessary
    
    filename = os.path.join(save_path, f"epoch_{epoch}_{int(y.detach())}.jpg")
    save_image(fake_image, filename, nrow=1, normalize=True)
    