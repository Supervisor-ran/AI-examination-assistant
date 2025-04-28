from PIL import Image
import os

# 设置文件夹路径
folder_path = 'data/'

# 创建输出文件夹
output_folder = 'data/processed_image'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
i = 1
# 遍历文件夹中的所有图像
for filename in os.listdir(folder_path):

    if filename.endswith('.jpg') or filename.endswith('.png'):  # 支持jpg和png文件
        img_path = os.path.join(folder_path, filename)
        with Image.open(img_path) as img:
            # 获取图像的宽和高
            width, height = img.size

            # 计算剪裁区域 (左, 上, 右, 下)
            left = 0
            top = max(0, height - 1694)
            right = width
            bottom = height

            # 剪裁图像
            cropped_img = img.crop((left, top, right, bottom))

            # 保存剪裁后的图像到输出文件夹
            cropped_img.save(os.path.join(output_folder, filename))
            print(f"Currently {i} images processed")
            i+=1
print("图像剪裁完成！")
