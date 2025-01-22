import torch
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
llm = OpenAI(api_key=OPENAI_API_KEY)

df = pd.read_csv("your_dataset_path")

# question과 llm_answer 겹치는 부분 제거
df['llm_answer'] = df.apply(
    lambda row: row['llm_answer'].replace(row['question'], ''),
    axis=1
)

########################################
# 1. GPT에게 정오답 판정 (Question/Gold/Answer/LLM 사용)
########################################
def judge_correctness_with_gold(
    question: str,
    gold_answer: str,
    answer: str,
    llm_answer: str
) -> str:
    """
    question, gold_answer, answer, llm_answer 네 가지 정보를 GPT에게 주고
    'correct' 또는 'incorrect'로만 판정받습니다.
    """

    chat_history = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "You are given the question, a gold (true) answer, and the LLM's response. "
                "Please judge if the LLM's response is correct. "
                "If the response is correct, respond exactly with 'correct'. "
                "If it's incorrect, respond exactly with 'incorrect'."
                "If the response is empty, respond exactly with 'incorrect'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Gold Answer: {gold_answer}\n"
                f"LLM Response: {llm_answer}\n\n"
                "Is the predicted answer correct?"
            )
        }
    ]

    response = llm.chat.completions.create(
        model="gpt-4o-mini",  # 예시 모델명, 실제 사용 모델로 수정
        messages=chat_history,
        temperature=0.7
    )
    final_response = response.choices[0].message.content.strip().lower()
    print("GPT 판단:", final_response)
    return final_response


########################################
# 2. df 에 대해 정오답 판정 컬럼 추가
########################################
def add_correctness_column_with_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    df 의 각 행에 대해:
      - question, gold_answer, answer, llm_answer 를 GPT에게 전달하여
        정오답 판정('correct' 또는 'incorrect') -> 'is_correct' 컬럼 생성
    """
    is_correct_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row.get('question', "")
        gold_answer = row.get('gold_answer', "")
        answer = row.get('answer', "")
        llm_answer = row.get('llm_answer', "")

        # GPT에게 판정받기
        judge_result = judge_correctness_with_gold(
            question=question,
            gold_answer=gold_answer,
            answer=answer,
            llm_answer=llm_answer
        )

        # judge_result 가 "correct" 또는 "incorrect" 로 온다고 가정
        if judge_result == "correct":
            is_correct_list.append("correct")
        else:
            # 그 외 응답은 모두 "incorrect"로 처리
            is_correct_list.append("incorrect")

    df['is_correct'] = is_correct_list
    return df

if __name__ == "__main__":

    # 정오답 평가
    df = add_correctness_column_with_gold(df)
