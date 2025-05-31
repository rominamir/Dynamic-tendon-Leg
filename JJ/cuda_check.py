import torch

print("✅ PyTorch version:", torch.__version__)
print("✅ CUDA version:", torch.version.cuda)
print("✅ CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("✅ Device name:", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3).to("cuda")
    print("✅ Test tensor on GPU:\n", x)
else:
    print("⚠️ Using CPU only.")
