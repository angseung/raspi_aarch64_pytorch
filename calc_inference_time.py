import time
import torch
from torchvision import models as models

x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
model = models.detection.ssdlite320_mobilenet_v3_large()
model.eval()

inference_time_total = 0.0
n_iters: int = 100

for _ in range(n_iters):
    start_time = time.time()
    dummy_output = model(x)
    end_time = time.time()

    inference_time_total += (end_time - start_time)

print(inference_time_total / n_iters)
