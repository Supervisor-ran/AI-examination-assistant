from flask import Flask, render_template, request, session
from func import *
from transformers import BertTokenizer,BertModel

app = Flask(__name__)
app.secret_key = 'llliiiuuurrrr'
# 初始化 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/generate', methods=['GET'])
def generate():
    # Multiple Choice Question
    mcq_with_options = [
        {
            'question': '后端应用开发的初期主要采用哪种模式？',
            'options': ['A. 微服务模式', 'B. 单体模式', 'C. 分布式系统', 'D. 服务器无关模式'],
            'answer': 'B'
        },
        {'question': '随着应用功能的复杂性增加，开发团队的规模如何变化？',
         'options': ['A. 一直保持不变', 'B. 减少', 'C. 增加', 'D. 不确定'],
         'answer': 'C'
         }

    ]

    # Fill-in-the-Blank Question
    fill_in_the_blank = [
        {'question': '单体模式的主要缺点是________。', 'answer': '项目迭代流程过于集中'},
        {'question': '微服务模式的目标是________各个服务。', 'answer': '解耦'}
    ]

    # Short Answer Question
    short_answer = [
        {
            'question': '为什么单体模式不适合大型复杂项目？',
            'answer': '因为单体模式使得项目迭代流程过于集中，导致开发效率降低，维护困难。'
        },
        {
            'question': '微服务模式有哪些优点？',
            'answer': '微服务模式通过解耦服务来提升灵活性，提高项目迭代效率，并可以独立部署和管理各个微服务。'
        }
    ]

    session['mcq_with_options'] = mcq_with_options
    session['fill_in_the_blank'] = fill_in_the_blank
    session['short_answer'] = short_answer

    return render_template('exam.html', mcq_with_options=mcq_with_options, fill_in_the_blank=fill_in_the_blank,
                           short_answer=short_answer)


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
        real_value = calculate_similarity(doc1,doc2, tokenizer, model)
        x = np.array([low_bound, real_value, 1])
        x_nor = normalize(x)
        if x_nor[1] > 0.95:
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


if __name__ == '__main__':
    app.run(debug=True)
