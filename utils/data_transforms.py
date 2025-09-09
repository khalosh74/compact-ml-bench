# utils/data_transforms.py
from torchvision import transforms

# Single source of truth (CIFAR-10)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def train_transform_cifar10():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def test_transform_cifar10():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

# For PTQ/QAT calibration; keep it simple/identical to test to avoid drift
def calib_transform_cifar10():
    return test_transform_cifar10()
