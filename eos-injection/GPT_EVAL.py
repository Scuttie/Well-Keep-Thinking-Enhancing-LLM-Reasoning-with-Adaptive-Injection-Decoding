import pandas as pd
from tqdm import tqdm
import openai
import os

def judge_correctness_with_api(row, llm):
    """
    GPT API를 사용하여 generated-answer와 answer 비교.
    """
    chat_history = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "The generated answer includes reasoning, and the answer is contained within the reasoning. "
                "Your task is to extract the answer from the reasoning and compare it to the correct answer. "
                "If they are identical or convey the same meaning, respond with 'correct'. "
                "If they are different, respond with 'wrong'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Correct Answer: {row['Answer']}\n"
                f"Generated Answer: {row['Generated-Answer']}\n\n"
                "Are the two answers identical or convey the same meaning?"
            )
        }
    ]

    response = llm.ChatCompletion.create(
        model="gpt-4o",
        messages=chat_history,
        temperature=0.7
    )
    return response.choices[0].message.content.strip().lower()


def evaluate_answers(input_csv, llm, output_csv=None):
    """
    CSV 파일에서 answer와 generated-answer 비교하여 정답 여부 판단.
    결과를 새로운 DataFrame으로 저장하고, 정답/오답 수와 목록 출력.
    """
    df = pd.read_csv(input_csv)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        correctness = judge_correctness_with_api(row, llm)
        results.append(correctness)

    df['Correctness'] = results

    correct_df = df[df['Correctness'] == 'correct']
    incorrect_df = df[df['Correctness'] == 'wrong']

    correct_count = len(correct_df)
    incorrect_count = len(incorrect_df)

    print(f"Correct: {correct_count}, Wrong: {incorrect_count}")
    print("\nCorrect Problems:\n", correct_df[['Question', 'Answer', 'Generated-Answer']])
    print("\nWrong Problems:\n", incorrect_df[['Question', 'Answer', 'Generated-Answer']])

    # 결과 저장 (선택 사항)
    if output_csv:
        df.to_csv(output_csv, index=False, encoding='utf-8')

    return df, correct_df, incorrect_df

def main():
    input_csv = "./results/Mistral-7B-v0.1/gsm8k/zero-shot promp + or_.csv"
    output_csv = "./results/Mistral-7B-v0.1/gsm8k/zero-shot promp + or__eval.csv"
    evaluate_answers(input_csv, openai, output_csv=None)
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # 평가 수행
    df, correct_df, incorrect_df = evaluate_answers(input_csv, openai, output_csv)

if __name__ == "__main__":
    main()