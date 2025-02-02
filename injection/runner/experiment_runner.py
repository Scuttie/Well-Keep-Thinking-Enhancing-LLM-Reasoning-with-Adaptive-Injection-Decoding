import re
import pandas as pd
from datasets import load_dataset
import os

from model_utils.generation_methods import (
    generate_greedy_decoding,
    generate_zero_shot_cot,
    generate_step_injection,
    generate_top_k_injection,
    generate_zs_next_step,
    generate_zs_top_k_multiple_injection_with_patience
)

from parser.parser_code import parse_txt_to_csv
from evaluator.correctness_judgement import add_correctness_column_with_gold

def run_experiment_for_method_commonsenseqa(
    model,
    tokenizer,
    dataset,
    method_name,
    output_txt_path,
    max_samples=100,
    max_length=500,
    top_k_value=10,
    injection_token="Step",
    patience_value=50,
    with_cot_init=False,
):
    """
    CommonsenseQA 전용:
      - dataset[i]['question'], dataset[i]['answer']
      - chain-of-thought 등은 따로 없으니 간단히 처리
    """
    def gen_fn(_prompt):
        if method_name == "greedy":
            return generate_greedy_decoding(model, tokenizer, _prompt, max_length)
        elif method_name == "zs_cot":
            return generate_zero_shot_cot(model, tokenizer, _prompt, max_length)
        elif method_name == "step_injection":
            return generate_step_injection(model, tokenizer, _prompt,
                                           max_length=max_length,
                                           injection_token=injection_token)
        elif method_name == "top_k_injection":
            return generate_top_k_injection(
                model, tokenizer, _prompt,
                k=top_k_value,
                injection_token=injection_token,
                max_length=max_length,
                with_cot_init=with_cot_init
            )
        elif method_name == "zs_next_step":
            return generate_zs_next_step(model, tokenizer, _prompt,
                                         max_length=max_length,
                                         injection_token=injection_token)
        elif method_name == "zs_top_k_multi_inject_patience":
            return generate_zs_top_k_multiple_injection_with_patience(
                model, tokenizer, _prompt,
                k=top_k_value,
                injection_token=injection_token,
                patience=patience_value,
                max_length=max_length
            )
        else:
            raise ValueError(f"Unknown method_name: {method_name}")

    # dataset은 이미 Dataset 형태 (train/test 분할 없음), len(dataset)
    num_data = len(dataset)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i in range(num_data):
            if i >= max_samples:
                break

            question_text = dataset[i]["question"]
            gold_answer = dataset[i]["answer"]

            f.write(f"\n{'='*28}\n")
            f.write(f"Sample #{i+1}\n")
            f.write(f"Question: {question_text}\n")

            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            # 모델 생성
            final_answer = gen_fn(question_text)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (CommonsenseQA) 결과 저장 완료 -> {output_txt_path}")


def run_experiment_for_method_gsm8k(
    model,
    tokenizer,
    dataset,
    method_name,
    output_txt_path,
    max_samples=100,
    max_length=500,
    top_k_value=10,
    injection_token="Step",
    patience_value=50
):
    """
    기존 GSM8K 스타일 (dataset['test']['question'], dataset['test']['answer']에서 분리)
    '####' 전후로 chain_of_thought / gold_answer 분리하는 방식
    """
    def gen_fn(_prompt):
        if method_name == "greedy":
            return generate_greedy_decoding(model, tokenizer, _prompt, max_length)
        elif method_name == "zs_cot":
            return generate_zero_shot_cot(model, tokenizer, _prompt, max_length)
        elif method_name == "step_injection":
            return generate_step_injection(model, tokenizer, _prompt,
                                           max_length=max_length,
                                           injection_token=injection_token)
        elif method_name == "top_k_injection":
            return generate_top_k_injection(
                model, tokenizer, _prompt,
                k=top_k_value,
                injection_token=injection_token,
                max_length=max_length
            )
        elif method_name == "zs_next_step":
            return generate_zs_next_step(model, tokenizer, _prompt,
                                         max_length=max_length,
                                         injection_token=injection_token)
        elif method_name == "zs_top_k_multi_inject_patience":
            return generate_zs_top_k_multiple_injection_with_patience(
                model, tokenizer, _prompt,
                k=top_k_value,
                injection_token=injection_token,
                patience=patience_value,
                max_length=max_length
            )
        else:
            raise ValueError(f"Unknown method_name: {method_name}")

    num_data = len(dataset['test']['question'])

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i, question_text in enumerate(dataset['test']['question']):
            if i >= max_samples:
                break

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

            final_answer = gen_fn(question_text)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (GSM8K) 결과 저장 완료 -> {output_txt_path}")


def run_experiment_for_method_math500(
    model,
    tokenizer,
    dataset,
    method_name,
    output_txt_path,
    max_samples=100,
    max_length=500,
    top_k_value=10,
    injection_token="Step",
    patience_value=50
):
    """
    MATH-500 전용
    """
    def gen_fn(_prompt):
        if method_name == "greedy":
            return generate_greedy_decoding(model, tokenizer, _prompt, max_length)
        elif method_name == "zs_cot":
            return generate_zero_shot_cot(model, tokenizer, _prompt, max_length)
        elif method_name == "step_injection":
            return generate_step_injection(model, tokenizer, _prompt,
                                           max_length=max_length,
                                           injection_token=injection_token)
        elif method_name == "top_k_injection":
            return generate_top_k_injection(
                model, tokenizer, _prompt,
                k=top_k_value,
                injection_token=injection_token,
                max_length=max_length
            )
        elif method_name == "zs_next_step":
            return generate_zs_next_step(model, tokenizer, _prompt,
                                         max_length=max_length,
                                         injection_token=injection_token)
        elif method_name == "zs_top_k_multi_inject_patience":
            return generate_zs_top_k_multiple_injection_with_patience(
                model, tokenizer, _prompt,
                k=top_k_value,
                injection_token=injection_token,
                patience=patience_value,
                max_length=max_length
            )
        else:
            raise ValueError(f"Unknown method_name: {method_name}")

    test_data = dataset['test']
    num_data = len(test_data)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i in range(num_data):
            if i >= max_samples:
                break

            question_text = test_data['problem'][i]
            chain_of_thought = test_data['answer'][i] if test_data['answer'][i] else ""
            solution_text = test_data['solution'][i] if test_data['solution'][i] else ""

            # \boxed{...} 추출
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

            final_answer = gen_fn(question_text)

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
    output_dir=".",
    top_k_value=10,
    injection_token="Step",
    patience_value=50,
    with_cot_init=False,
):
    """
    전체 파이프라인 실행:
      - (모델, method)에 대해 실험 수행 -> txt 생성
      - txt -> csv 변환
      - GPT(OpenAI)로 정오답 판정 -> csv 컬럼 추가
    """
    os.makedirs(output_dir, exist_ok=True)

    from parser.parser_code import parse_txt_to_csv

    # 어떤 전용 함수를 쓸지 분기
    is_math500 = "math-500" in dataset_name.lower()
    is_commonsense = "commonsenseqa" in dataset_name.lower()
    # 그 외 GSM8K류

    for method_name in methods:
        output_txt_path = f"{output_dir}/{model_name}_{method_name}.txt"

        if is_math500:
            run_experiment_for_method_math500(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                method_name=method_name,
                output_txt_path=output_txt_path,
                max_samples=max_samples,
                max_length=max_length,
                top_k_value=top_k_value,
                injection_token=injection_token,
                patience_value=patience_value
            )
        elif is_commonsense:
            run_experiment_for_method_commonsenseqa(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                method_name=method_name,
                output_txt_path=output_txt_path,
                max_samples=max_samples,
                max_length=max_length,
                top_k_value=top_k_value,
                injection_token=injection_token,
                patience_value=patience_value,
                with_cot_init=with_cot_init
            )
        else:
            # GSM8K 스타일
            run_experiment_for_method_gsm8k(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                method_name=method_name,
                output_txt_path=output_txt_path,
                max_samples=max_samples,
                max_length=max_length,
                top_k_value=top_k_value,
                injection_token=injection_token,
                patience_value=patience_value
            )

        # txt -> csv
        output_csv_path = f"{output_dir}/{model_name}_{method_name}.csv"
        parse_txt_to_csv(output_txt_path, output_csv_path)

        # GPT(OpenAI) 평가
        if llm is not None:
            df = pd.read_csv(output_csv_path)
            df_eval = add_correctness_column_with_gold(df, llm=llm)
            final_csv_path = f"{output_dir}/{model_name}_{method_name}_evaluated.csv"
            df_eval.to_csv(final_csv_path, index=False, encoding='utf-8')
            print(f"Final Evaluation CSV saved -> {final_csv_path}")
        else:
            print("llm(None) -> GPT 평가 스킵")

    print("\n[모든 실험 완료]")
