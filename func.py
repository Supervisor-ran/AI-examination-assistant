import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
import random
from openai import OpenAI
import time
import torch
import numpy as np


# 设置日志配置
def initilize_log():
    # 创建一个 RotatingFileHandler 对象
    log_path = 'logs/'
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    handler = RotatingFileHandler(os.path.join(log_path, 'AI_review_sys_log.txt'), maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)

    # 创建一个 formatter 对象并设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    return handler


def unix_to_formatted_time():
    t = time.time()
    # 将 Unix 时间戳转换为 datetime 对象
    dt = datetime.fromtimestamp(t)

    # 将毫秒部分提取出来
    milliseconds = int((t % 1) * 1000)

    # 格式化 datetime 对象为所需的格式
    formatted_time = dt.strftime(f'%y%m%d%H%M%S{milliseconds:03d}')

    return formatted_time


# 读取文件内容并管理成JSON格式
def read_files(folder_path, filenames):
    file_contents = {}
    file_weights = {}

    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        # abs_path = os.path.abspath(file_path)
        # print(f"Absolute path: {abs_path}")  # 打印绝对路径
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_contents[filename] = content
                file_weights[filename] = 1 / len(filenames)  # 初始权重设为1，之后可以根据需要调整
        except FileNotFoundError:
            print(f"File not found: {file_path}")

    return file_contents, file_weights


# 根据权重随机选择文件并从中选择段落
def select_weighted_content(file_contents, file_weights, n_files=None, n_paragraphs=None):
    # 使用权重随机选择文件
    selected_files = random.choices(
        population=list(file_contents.keys()),
        weights=list(file_weights.values()),
        k=n_files
    )

    # 去除重复的文件
    selected_files = list(set(selected_files))

    # 如果选择的文件少于n_files，则补足
    while len(selected_files) < n_files:
        if len(list(file_contents.keys())) < n_files:
            break
        additional_files = random.choices(
            population=list(file_contents.keys()),
            weights=file_weights,
            k=n_files - len(selected_files)
        )
        selected_files.extend(additional_files)
        selected_files = list(set(selected_files))  # 去重

    # 合并选中文件的内容
    merged_content = ""
    for file in selected_files:
        merged_content += file_contents[file] + "\n\n"

    # 分割成段落
    paragraphs = merged_content.split("\n\n")

    # 如果段落数不足n_paragraphs，则调整
    if len(paragraphs) < n_paragraphs:
        n_paragraphs = len(paragraphs)

    # 随机选择指定数量的段落
    selected_paragraphs = random.sample(paragraphs, n_paragraphs)

    return "\n\n".join(selected_paragraphs)


def generate_questions(content):
    client = OpenAI()
    prompt = f"""
    根据以下材料，生成20个单项选择题，10个填空题，5个简答题。为所有问题提供答案。单项选择题的答案应包含四个选项（A, B, C, D）。
    材料：
    {content}
    请将输出格式化为JSON，结构如下：
    {{
        "multiple_choice": [
            {{
                "question": "题目内容",
                "options": ["A.选项", "B.选项", "C.选项", "D.选项"],
                "answer": "正确选项的字母"
            }},
            ...
        ],
        "fill_in_the_blank": [
            {{
                "question": "题目内容",
                "answer": "正确答案"
            }},
            ...
        ],
        "short_answer": [
            {{
                "question": "题目内容",
                "answer": "正确答案"
            }},
            ...
        ]
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 或使用适合的模型
        messages=[
            {"role": "system", "content": "你是一个助理，帮助生成考试题目。"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens


def save_text_to_file(text, folder_path, file_name):
    """
    将文本保存到指定目录的文件中。

    :param text: 要保存的文本内容
    :param folder_path: 文件夹路径
    :param file_name: 文件名
    """
    # 确保目录存在
    os.makedirs(folder_path, exist_ok=True)

    # 拼接文件路径
    file_path = os.path.join(folder_path, file_name)

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"文本成功保存到 {file_path}")
    except IOError as e:
        print(f"保存文本时出错: {e}")


def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def calculate_similarity(text1, text2, tokenizer, model):
    embedding1 = get_bert_embedding(text1, tokenizer, model)
    print(embedding1)
    embedding2 = get_bert_embedding(text2, tokenizer, model)

    norm1 = np.linalg.norm(embedding1)
    print(norm1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return similarity


def normalize(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


if __name__ == '__main__':
    # app.run(debug=True)
    # response = generate()
    # print(response)
    from transformers import BertTokenizer, BertModel

    doc1 = "我喜欢单体模式"
    doc2 = "在单体模式下系统解耦性低，并行开发效率降低，后续维护也困难"
    doc3 = "因为单体模式使得项目迭代流程过于集中，导致开发效率降低，维护困难。"
    doc4 = ""
    doc5 = "因为单体模式使得项目迭代流程过于集中，导致开发效率降低，维护困。"
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    similarity1 = calculate_similarity(doc1, doc3, tokenizer, model)
    similarity2 = calculate_similarity(doc5, doc3, tokenizer, model)
    similarity3 = calculate_similarity(doc4, doc3, tokenizer, model)

    print(similarity1, similarity2, similarity3)

    import math

    print(math.log(similarity1), math.log(similarity2), math.log(similarity3))
    print(3)
