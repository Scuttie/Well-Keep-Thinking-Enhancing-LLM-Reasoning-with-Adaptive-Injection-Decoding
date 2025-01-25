import re
import pandas as pd
from datasets import load_dataset

from model_utils.generation_methods import (
    generate_greedy_decoding,
    generate_zero_shot_cot,
    generate_step_injection,
    generate_top_k_injection,
    generate_zs_next_step
)

from parser.parser_code import parse_txt_to_csv
from evaluator.correctness_judgement import add_correctness_column_with_gold


########################################
# 2. (모델, Method)에 대해 실험 수행 -> txt 저장
########################################
def run_experiment_for_method(
    model,
    tokenizer,
    dataset,
    method_name,
    output_txt_path,
    max_samples=100,
    max_length=500
):
    """
    주어진 (model, tokenizer)에 대해 dataset 상위 max_samples개를
    method_name에 해당하는 방식으로 디코딩 후, output_txt_path에 결과 저장.
    (GSM8K 등 question/answer 형식)
    """
    generation_fn_map = {
        "greedy": generate_greedy_decoding,
        "zs_cot": generate_zero_shot_cot,
        "step_injection": generate_step_injection,
        "top_k_injection": generate_top_k_injection,
        "zs_next_step": generate_zs_next_step
    }

    gen_fn = generation_fn_map[method_name]

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i, question_text in enumerate(dataset['test']['question']):
            if i >= max_samples:
                break

            # 모범답안/정답 분리
            cot_with_answer = dataset['test']['answer'][i]
            splitted = cot_with_answer.split("####")
            chain_of_thought = splitted[0].strip() if len(splitted) > 0 else ""
            gold_answer = splitted[1].strip() if len(splitted) > 1 else ""

            f.write(f"\n{'='*28}\n")
            f.write(f"Test sample #{i+1}\n")
            f.write(f"Question: {question_text}\n")

            f.write("\n[모범답안]\n")
            f.write(chain_of_thought + "\n")
            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            final_answer = gen_fn(model, tokenizer, question_text, max_length=max_length)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] 결과 저장 완료 -> {output_txt_path}")


def run_experiment_for_method_math500(
    model,
    tokenizer,
    dataset,
    method_name,
    output_txt_path,
    max_samples=100,
    max_length=500
):
    """
    MATH-500 전용 함수:
      - dataset['test']에는 "problem", "answer", "solution" 컬럼이 있다고 가정.
      - problem -> Question
      - answer -> 모범답안(Chain-of-Thought)
      - solution -> 골드 정답이 들어 있는 문자열(\\boxed{...} 등)
    """
    generation_fn_map = {
        "greedy": generate_greedy_decoding,
        "zs_cot": generate_zero_shot_cot,
        "step_injection": generate_step_injection,
        "top_k_injection": generate_top_k_injection,
        "zs_next_step": generate_zs_next_step
    }
    gen_fn = generation_fn_map[method_name]

    test_data = dataset['test']
    num_data = len(test_data)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i in range(num_data):
            if i >= max_samples:
                break

            # 1) Question
            question_text = test_data['problem'][i]

            # 2) 모범답안(Chain-of-Thought)
            chain_of_thought = test_data['answer'][i] if test_data['answer'][i] else ""

            # 3) gold answer 추출 (\boxed{...})
            solution_text = test_data['solution'][i] if test_data['solution'][i] else ""
            matches = re.findall(r'\\boxed\{(.*?)\}', solution_text)
            if matches:
                gold_answer = matches[-1].strip()
            else:
                gold_answer = "N/A"

            f.write(f"\n{'='*28}\n")
            f.write(f"Test sample #{i+1}\n")
            f.write(f"Question: {question_text}\n")

            f.write("\n[모범답안]\n")
            f.write(chain_of_thought + "\n")
            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            final_answer = gen_fn(model, tokenizer, question_text, max_length=max_length)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (MATH-500) 결과 저장 완료 -> {output_txt_path}")


def run_full_pipeline(
    llm,
    model,
    tokenizer,
    dataset_name,
    dataset,
    model_name,
    methods,
    max_samples=100,
    max_length=500,
    output_dir="."
):
    """
    전체 파이프라인 실행:
      - (모델, method)에 대해 실험 수행 -> txt 생성
      - txt -> csv 변환
      - GPT(OpenAI)로 정오답 판정 -> csv 컬럼 추가
    """
    # dataset_name 에 따라, GSM8K 스타일인지 MATH-500 스타일인지 분기
    # (여기서는 'MATH-500'이라는 키워드로만 단순하게 구분 예시)
    is_math500 = "math-500" in dataset_name.lower()

    from parser.parser_code import parse_txt_to_csv

    for method_name in methods:
        # txt
        output_txt_path = f"{output_dir}/{model_name}_{method_name}.txt"
        if is_math500:
            run_experiment_for_method_math500(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                method_name=method_name,
                output_txt_path=output_txt_path,
                max_samples=max_samples,
                max_length=max_length
            )
        else:
            run_experiment_for_method(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                method_name=method_name,
                output_txt_path=output_txt_path,
                max_samples=max_samples,
                max_length=max_length
            )

        # txt -> csv
        output_csv_path = f"{output_dir}/{model_name}_{method_name}.csv"
        parse_txt_to_csv(output_txt_path, output_csv_path)

        # GPT(OpenAI) 평가
        df = pd.read_csv(output_csv_path)
        df_eval = add_correctness_column_with_gold(df, llm=llm)  # GPT 평가
        final_csv_path = f"{output_dir}/{model_name}_{method_name}_evaluated.csv"
        df_eval.to_csv(final_csv_path, index=False, encoding='utf-8')
        print(f"Final Evaluation CSV saved -> {final_csv_path}")

    print("\n[모든 실험 완료]")
