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
    
    print(f"summaries {len(summaries)}, questions {len(questions)}")

    # 添加 例子
    examples = []
    for i, (summary, que) in enumerate(zip(summaries, questions)):
        examples.append([summary, que])

    num_of_examples = len(examples)

    model_name = "ERNIE-4.0-Turbo-8K"
    print(f"model_name = {model_name}")

    clean_Q_A = []

    dataset_test = pd.read_excel("XXXXXX.xlsx")
    summaries_test, questions_test = dataset_test["报告_sum"], dataset_test["考核-提问"]

    for i, (summary, que) in tqdm(enumerate(zip(summaries_test, questions_test)), total=len(summaries)):
        rand_i = random.randrange(0, num_of_examples)
        examp_sum = examples[rand_i][0]
        examp_que = examples[rand_i][1]

        system_content = f"你是一个资深临床医学专家" # content (system)
        user_content = f" # 任务：参考以下报告样例与问题样例，为患者报告写一个提问。\n # 报告样例：{examp_sum}，\n # 问题样例：{examp_que}，\n # 患者报告：{summary}，\n # 输出格式:\n \"提问\": \"\"" # content (user)

        model_output = chat_comp.do(model=model_name, 
            messages=[
                {
                    "role": "user",
                    "content": f"{system_content}{user_content}"
                },
            ]
        )
        model_output = model_output["body"]["result"]