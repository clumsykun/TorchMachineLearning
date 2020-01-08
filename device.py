from torch.cuda import is_available

if is_available():
    DEVICE = "cuda:0"
else:
    raise ValueError("GPU 不可用！")