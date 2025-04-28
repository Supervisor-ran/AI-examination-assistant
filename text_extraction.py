import pytesseract
import time
import os

process_id = time.time()
# 设置 Tesseract 可执行文件的路径
pytesseract.pytesseract.tesseract_cmd = r'G:\tesseract\tesseract.exe'


def ocr_from_image(image_path):
    from PIL import Image
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='chi_sim')  # 使用简体中文进行 OCR
    return text


def ocr_from_images(image_folder, save_indices):
    all_text = ""
    filenames = [f"SAD_material_{i}.png" for i in range(676, 701)]
    chapter = 2

    for filename in filenames:
        # 提取文件名中的数字部分
        index = int(filename.split('_')[2].split('.')[0])

        # 构造完整的图片路径
        image_path = os.path.join(image_folder, filename)
        text = ocr_from_image(image_path)

        # 累积文本
        all_text += text + "\n"

        print(f"Current processing file is {filename}")

        # 检查当前索引是否在保存索引数组中
        if index in save_indices:
            # 保存当前累积的文本
            all_text = all_text.replace(" ", "")
            with open(f"extracted_text/extracted_text_chapter{19}_{int(process_id)}.txt", 'w',
                      encoding='utf-8') as file:
                file.write(all_text)
            print(f"第{chapter}章保存完毕。")
            chapter += 1

            # 清空累积文本
            all_text = ""

    return all_text


# 定义要保存的索引列表
save_indices = [104, 144, 174, 217, 247,
                270, 304, 329, 368, 404,
                450, 481, 511, 540, 598,
                632, 675, 700]  # 你可以将这里的值替换为你需要的索引

# 运行 OCR 处理并保存文本
text = ocr_from_images("data/processed_image", save_indices)
