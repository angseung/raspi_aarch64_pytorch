import torch
from torchvision import transforms

image = torch.randn((1, 3, 224, 224), requires_grad=True)
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(image)
# torch.backends.quantized.engine = 'qnnpack'

from torchvision import models
net = models.mobilenet_v2(weights=None)

#net = torch.jit.script(net)

#with torch.no_grad():
net.train()
output = net(image)

output_scala = torch.sum(output, dim=1)
output_scala.backward()
print(output_scala.shape)

print(output.shape)
