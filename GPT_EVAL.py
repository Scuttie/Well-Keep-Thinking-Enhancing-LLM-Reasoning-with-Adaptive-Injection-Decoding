import torch
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

def judge_correctness_without_gold(pred: str, llm) -> str:
    """
    gold 없이, 오직 pred(answer)만 보고
    GPT에게 'correct' 또는 'wrong' 판정을 요청합니다.
    """

    chat_history = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "You are given only a predicted answer, without the original question. "
                "Please judge if this answer is correct or not. "
                "If the predicted answer is correct, respond exactly with the word 'correct'. "
                "If it's incorrect, respond exactly with the word 'wrong'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Predicted Answer: {pred}\n\n"
                "Is the predicted answer correct?"
            )
        }
    ]

    response = llm.chat.completions.create(
        model="gpt-4o-mini",  
        messages=chat_history,
        temperature=0.7
    )
    final_response = response.choices[0].message.content.strip().lower()
    print("GPT 판단:", final_response)
    return final_response


def add_correctness_column_no_gold(df: pd.DataFrame, llm) -> pd.DataFrame:
    """
    df 의 각 행에 대해:
      - 'answer' 컬럼만을 GPT에게 전달하여 정오답 판정 -> 'is_correct' 컬럼 생성
    """
    is_correct_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 오직 answer만 사용 (question, reasoning 등은 사용 안 함)
        pred = row['answer']

        # GPT에게 판정받기
        judge_result = judge_correctness_without_gold(pred=pred, llm=llm)

        # judge_result 가 "correct" 또는 "wrong" 으로 온다고 가정
        if judge_result == "correct":
            is_correct_list.append("correct")
        else:
            is_correct_list.append("incorrect")

    df['is_correct'] = is_correct_list
    return df


