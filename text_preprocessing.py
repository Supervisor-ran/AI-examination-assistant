import os
import re

def process_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
                content = file.read()

            # 匹配所有二级标题的部分
            sections = re.split(r'(\n\d+\.\d+)', content)

            processed_content = ""
            for i in range(1, len(sections), 2):  # 遍历每个二级标题及其内容
                title = sections[i].strip()
                content = sections[i + 1].replace('\n', ' ').strip()
                processed_content += f"{title} {content}\n\n"

            # 保存处理后的内容到输出文件夹中
            output_filename = os.path.join(output_folder, filename)
            with open(output_filename, 'w', encoding='utf-8') as output_file:
                output_file.write(processed_content)

input_folder = 'data/process1'   # 替换为你的输入文件夹路径
output_folder = 'data/process2' # 替换为你的输出文件夹路径

process_files(input_folder, output_folder)
