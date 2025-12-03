# encoding=utf-8

import jieba
import pandas as pd
import random
import sys
sys.path.append(".")

import pandas as pd
import time
from tqdm import tqdm
import json

import os
import qianfan

os.environ["QIANFAN_ACCESS_KEY"] = "XXXXXX"
os.environ["QIANFAN_SECRET_KEY"] = "XXXXXX"
chat_comp = qianfan.ChatCompletion()
if __name__ == "__main__":
    random.seed(42)

    dataset = pd.read_excel("XXXXXX.xlsx")
    summaries = dataset["报告_sum"]
    questions = dataset["考核-提问"]
    answers = dataset["考核-答案"]
    reviews = dataset["考核-答案"]
    
    print(f"summaries {len(summaries)}, questions {len(questions)}")

    # 添加 例子
    examples = []
    for i, (summary, que, ans, rev) in enumerate(zip(summaries, questions, answers, reviews)):
        examples.append([summary, que, ans, rev])

    num_of_examples = len(examples)

    model_name = "ERNIE-4.0-Turbo-8K"
    print(f"model_name = {model_name}")

    clean_Q_A = []

    dataset_test = pd.read_excel("XXXXXX.xlsx")
    summaries_test, questions_test, answers_test = dataset_test["报告_sum"], dataset_test["考核-提问"]

    for i, (summary, que, ans) in tqdm(enumerate(zip(summaries_test, questions_test, answers_test)), total=len(summaries)):
        time_0 = time.time()
        rand_i = random.randrange(0, num_of_examples)
        examp_sum = examples[rand_i][0]
        examp_que = examples[rand_i][1]
        examp_ans = examples[rand_i][2]
        examp_rev = examples[rand_i][3]

        ref_rev = f"# 样例问题: {examp_que} \n # 样例回答: {examp_ans} \n # 答案评价: {examp_rev}"

        system_content = f"你是一个资深临床医学专家" # content (system)
        user_content = f" # 任务：根据医学问题和回答，检查是否需要纠正答案。如果\"是\"，给出修正的答案。如果\"否\"，答案留空。\n {ref_rev}\n\n# 医学问题：{que}， \n # 回答：{ans}，\n # 输出格式:\n \"是否需要纠正\": \"\", \n \"答案\": \"\"" # content (user)

        model_output = chat_comp.do(model=model_name, 
            messages=[
                {
                    "role": "user",
                    "content": f"{system_content}{user_content}"
                },
            ]
        )
        model_output = model_output["body"]["result"]
        