from PIL import Image
from torchvision import transforms

# 加载图像
image = Image.open('~/ling/BreaKHis_v1/histology_slides/breast')

# 定义 transform
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # #transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(32),  # 调整图像大小为 256x256 像素
        transforms.CenterCrop(32),  # 中心裁剪为 224x224 像素
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 应用 transform
transformed_image = transform(image)
image_size = transformed_image.shape[-3:]  # 获取图像的高度和宽度

print(image_size)

# 展示转换后的图像
import matplotlib.pyplot as plt

# Permute the dimensions of the image tensor to correctly display it
plt.imshow(transformed_image.permute(1, 2, 0))
plt.show()
