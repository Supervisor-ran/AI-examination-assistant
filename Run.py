import json
import re

from flask import Flask, render_template, request, session
from transformers import BertTokenizer,BertModel

from func import *

app = Flask(__name__)
app.secret_key = 'llliiiuuurrrr'
# 初始化 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

logger = logging.getLogger('AI review System')
logger.setLevel(logging.DEBUG)  # 设置日志记录器的最低级别
handler = None
if not app.debug:
    handler = initilize_log()
logger.addHandler(handler)

folder_path = "data/final/"  # 存放章节文件的文件夹
filenames = [f"chapter_{i}.txt" for i in range(2, 20)]
n_files = 2
n_para = 10


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/generate', methods=['GET'])
def generate():
    app.logger.info('Collecting materials.')
    # generate input
    file_contents, file_weights = read_files(folder_path, filenames)
    content = select_weighted_content(file_contents, file_weights, n_files, n_para)
    app.logger.info(f'Materials were collected.')

    # generate problem
    app.logger.info(f'Starting to generate problems')
    questions_str, input_tokens, output_tokens = generate_questions(content)
    # 使用正则表达式提取JSON部分
    json_match = re.search(r'```json\n(.*?)\n```', questions_str, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
        # 解析JSON字符串为Python字典
        output_data = json.loads(json_str)
    else:
        raise Exception("没有找到有效的JSON内容")

    # save input
    start_time = unix_to_formatted_time()
    content = f"This input was generated at {start_time}, and token length is {input_tokens}." + \
              '\n\n' + content
    save_text_to_file(content, 'input/', f'input_{start_time}.txt')
    app.logger.info(f'Input data were saved.')

    # save output
    end_time = unix_to_formatted_time()
    output_data_str = json.dumps(output_data,indent=4)
    output_content = f"This output was generated at {end_time}, and token length is {output_tokens}." + \
                     '\n\n' + output_data_str
    save_text_to_file(output_content, 'output/', f'output_{end_time}.txt')

    session['mcq_with_options'] = output_data['multiple_choice']
    session['fill_in_the_blank'] = output_data['fill_in_the_blank']
    session['short_answer'] = output_data['short_answer']

    return render_template('exam.html', mcq_with_options=output_data['multiple_choice'],
                           fill_in_the_blank=output_data['fill_in_the_blank'], short_answer=output_data['short_answer'])


@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve questions from session
    mcq_with_options = session.get('mcq_with_options', [])
    fill_in_the_blank = session.get('fill_in_the_blank', [])
    short_answer = session.get('short_answer', [])

    user_answers = {
        'mcq': {q['question']: request.form.get(q['question']).split('.')[0].strip() if request.form.get(
            q['question']) else '' for q in mcq_with_options},
        'fill_in_the_blank': {q['question']: request.form.get(q['question']) for q in fill_in_the_blank},
        'short_answer': {q['question']: request.form.get(q['question']) for q in short_answer}
    }


    # 评分逻辑
    score = {'mcq': 0, 'fill_in_the_blank': 0, 'short_answer': 0}
    total_score = 0
    for item in mcq_with_options:
        if user_answers['mcq'].get(item['question']) == item['answer']:
            score['mcq'] += 1.5  # 假设每题分数为1分
    total_score += score['mcq']

    score_fill_in_blank = {}
    for item in fill_in_the_blank:
        doc1 = user_answers['fill_in_the_blank'].get(item['question'])
        doc2 = item['answer']
        doc3 = ""
        low_bound = calculate_similarity(doc3, doc2, tokenizer, model)
        real_value = calculate_similarity(doc1, doc2, tokenizer, model)
        x = np.array([low_bound, real_value, 1])
        x_nor = normalize(x)
        if x_nor[1] > 0.9:
            score_fill_in_blank[item['question']] = True
            score['fill_in_the_blank'] += 3
        else:
            score_fill_in_blank[item['question']] = False
    total_score += score['fill_in_the_blank']

    score_short_answers = {}
    for item in short_answer:
        doc1 = user_answers['short_answer'].get(item['question'])
        doc2 = item['answer']
        doc3 = ""
        low_bound = calculate_similarity(doc3, doc2, tokenizer, model)
        real_value = calculate_similarity(doc1, doc2, tokenizer, model)
        x = np.array([low_bound, real_value, 1])
        x_nor = normalize(x)
        score['short_answer'] += 6 * x_nor[1]
        score_short_answers[item['question']] = x_nor[1]
    total_score += score['short_answer']

    return render_template('result.html', user_answers=user_answers, mcq_with_options=mcq_with_options,
                           fill_in_the_blank=fill_in_the_blank, short_answer=short_answer, score=score,
                           total_score=total_score, score_fill_in_blank=score_fill_in_blank,
                           score_short_answers=score_short_answers)


if __name__ == "__main__":
    app.run(debug=True)
