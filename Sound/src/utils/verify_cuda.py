import torch
print(torch.__version__)
print(torch.version.cuda) # type: ignore # runtime linked
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Make a tensor on GPU
x = torch.rand(3, 3, device="cuda")
print(x)

# Simple operation on GPU
y = x @ x
print(y)

print("Tensor device:", y.device)

"""
Expected output:
2.4.0
12.4
90100
True
NVIDIA GeForce RTX 4060 Laptop GPU
tensor([[0.5499, 0.1550, 0.0748],
        [0.8519, 0.2266, 0.3988],
        [0.8710, 0.6051, 0.9296]], device='cuda:0')
tensor([[0.4996, 0.1656, 0.1725],
        [1.0089, 0.4247, 0.5248],
        [1.8041, 0.8346, 1.1706]], device='cuda:0')
Tensor device: cuda:0
"""
