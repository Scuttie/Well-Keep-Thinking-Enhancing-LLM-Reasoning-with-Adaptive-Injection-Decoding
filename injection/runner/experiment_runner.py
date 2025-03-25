import re
import pandas as pd
from datasets import load_dataset
import time
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
    '####' 기준으로 chain_of_thought / gold_answer 분리
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
    patience_value=50,
    with_cot_init=False
):
    """
    MATH-500 전용 (problem, answer만 존재)
    """
    # 메서드별 생성 함수 정의
    def gen_fn(_prompt):
        if method_name == "greedy":
            return generate_greedy_decoding(model, tokenizer, _prompt, max_length)
        elif method_name == "zs_cot":
            return generate_zero_shot_cot(model, tokenizer, _prompt, max_length)
        elif method_name == "step_injection":
            return generate_step_injection(
                model, tokenizer, _prompt,
                max_length=max_length,
                injection_token=injection_token
            )
        elif method_name == "top_k_injection":
            return generate_top_k_injection(
                model, tokenizer, _prompt,
                k=top_k_value,
                injection_token=injection_token,
                max_length=max_length,
                with_cot_init=with_cot_init
            )
        elif method_name == "zs_next_step":
            return generate_zs_next_step(
                model, tokenizer, _prompt,
                max_length=max_length,
                injection_token=injection_token
            )
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

            # 문제와 정답(전체 풀이 포함)을 가져옴
            question_text = test_data['problem'][i]
            full_answer_text = test_data['answer'][i] if test_data['answer'][i] else ""

            # \boxed{...} 패턴 추출
            matches = re.findall(r'\\boxed\{(.*?)\}', full_answer_text)
            if matches:
                gold_answer = matches[-1].strip()  # 마지막 \boxed{ }만 사용
            else:
                gold_answer = "N/A"

            # 로그 기록
            f.write(f"\n{'='*28}\n")
            f.write(f"Test sample #{i+1}\n")
            f.write(f"Question: {question_text}\n")

            # 모범답안(실제 정답 풀이)
            f.write("\n[모범답안]\n")
            f.write(full_answer_text + "\n")

            # 정답 (추출된 \boxed{})
            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            # 모델 추론
            final_answer = gen_fn(question_text)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (MATH-500) 결과 저장 완료 -> {output_txt_path}")

def run_experiment_for_method_lastletterconcat(
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
    with_cot_init=False
):
    """
    Last Letter Concatenation 전용:
      - 문제: dataset[i]['question']
      - 정답: dataset[i]['answer']
    """
    from model_utils.generation_methods import (
        generate_greedy_decoding,
        generate_zero_shot_cot,
        generate_step_injection,
        generate_top_k_injection,
        generate_zs_next_step,
        generate_zs_top_k_multiple_injection_with_patience
    )

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

            final_answer = gen_fn(question_text)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (LastLetterConcat) 결과 저장 완료 -> {output_txt_path}")

def run_experiment_for_method_strategyqa(
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
    with_cot_init=False
):
    """
    StrategyQA 전용:
      - 문제: dataset[i]['question']
      - 정답: dataset[i]['answer']
    """
    from model_utils.generation_methods import (
        generate_greedy_decoding,
        generate_zero_shot_cot,
        generate_step_injection,
        generate_top_k_injection,
        generate_zs_next_step,
        generate_zs_top_k_multiple_injection_with_patience
    )

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

            final_answer = gen_fn(question_text)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (StrategyQA) 결과 저장 완료 -> {output_txt_path}")


def run_experiment_for_method_aqua(
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
    with_cot_init=False
):
    """
    deepmind/aqua_rat 전용:
      - dataset['train' or 'test'] 등으로 분할 가능. (기본적으로 'train', 'test'가 존재)
      - 각 샘플 구조: 
         question (string), options (string), correct (string), rationale (string) ...
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

    ds = dataset['test']
    num_data = len(ds)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i in range(num_data):
            if i >= max_samples:
                break

            question_text = ds[i]["question"]
            options_text = ds[i]["options"]
            gold_answer = ds[i]["correct"]  # 정답 레이블 (ex: 'B')

            # Prompt 구성
            prompt = f"{question_text}\nOptions: {options_text}\n"

            f.write(f"\n{'='*28}\n")
            f.write(f"Test sample #{i+1}\n")
            f.write(f"Question:\n{question_text}\n")
            f.write(f"Options:\n{options_text}\n")

            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            final_answer = gen_fn(prompt)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (AQUA) 결과 저장 완료 -> {output_txt_path}")

def run_experiment_for_method_multiarith(
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
    token_pool=None,
    random_seed=1
):
    """
    MultiArith 전용
    """
    import random

    ADDITION_POOL = [
        "and", "so", "therefore", "then",
        "thus", "or", "in addition", "furthermore"
    ]
    CONTRAST_POOL = [
        "however", "but", "on the other hand", "yet",
        "in contrast", "nevertheless", "unlike", "instead", "conversely"
    ]
    MIX_POOL = ADDITION_POOL + CONTRAST_POOL

    def generate_top_k_injection_random(_prompt, idx):
        random.seed(random_seed + idx)

        if token_pool == "addition":
            chosen_token = random.choice(ADDITION_POOL)
        elif token_pool == "contrast":
            chosen_token = random.choice(CONTRAST_POOL)
        elif token_pool == "mix":
            chosen_token = random.choice(MIX_POOL)
        else:
            chosen_token = injection_token

        return generate_top_k_injection(
            model, tokenizer, _prompt,
            k=top_k_value,
            injection_token=chosen_token,
            max_length=max_length,
            with_cot_init=with_cot_init
        )

    def gen_fn(_prompt, idx):
        if method_name == "greedy":
            return generate_greedy_decoding(model, tokenizer, _prompt, max_length)
        elif method_name == "zs_cot":
            return generate_zero_shot_cot(model, tokenizer, _prompt, max_length)
        elif method_name == "step_injection":
            return generate_step_injection(model, tokenizer, _prompt,
                                           max_length=max_length,
                                           injection_token=injection_token)
        elif method_name == "top_k_injection":
            return generate_top_k_injection_random(_prompt, idx)
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

    num_data = len(dataset)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i in range(num_data):
            if i >= max_samples:
                break

            question_text = dataset[i]["question"]
            gold_answer = dataset[i]["final_ans"]

            f.write(f"\n{'='*28}\n")
            f.write(f"Sample #{i+1}\n")
            f.write(f"Question: {question_text}\n")

            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            start_time = time.time()
            final_answer = gen_fn(question_text, i)
            end_time = time.time()
            elapsed_time = end_time - start_time

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"\nTime taken: {elapsed_time:.4f}\n")

            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (MultiArith) 결과 저장 완료 -> {output_txt_path}")

def run_experiment_for_bbh(
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
    with_cot_init=False
):
    """
    BBH 전용 (예: "lukaemon/bbh"의 "date_understanding" 등)
      - dataset['test'] 안에 다 들어있다고 가정
      - 각 샘플:
          "input"  -> 문제/질문
          "target" -> 정답
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

    data_split = dataset["test"]  # bbh/date_understanding는 'test' split에 저장되어 있음
    num_data = len(data_split)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i in range(num_data):
            if i >= max_samples:
                break

            question_text = data_split[i]["input"]
            gold_answer = data_split[i]["target"]

            f.write(f"\n{'='*28}\n")
            f.write(f"Sample #{i+1}\n")
            f.write(f"Question: {question_text}\n")

            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            final_answer = gen_fn(question_text)

            f.write("\n--- [A] final answer ---\n")
            f.write(final_answer + "\n")
            f.write(f"{'='*28}\n\n")

    print(f"[{method_name}] (BBH) 결과 저장 완료 -> {output_txt_path}")


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
    token_pool=None,
    random_seed=1
):
    import os
    import pandas as pd
    from parser.parser_code import parse_txt_to_csv
    from evaluator.correctness_judgement import add_correctness_column_with_gold

    os.makedirs(output_dir, exist_ok=True)

    # 데이터셋 이름에 따른 분기
    is_math500 = "math-500" in dataset_name.lower()
    is_commonsense = "commonsenseqa" in dataset_name.lower()
    is_aqua = "aqua_rat" in dataset_name.lower() or "aqua-rat" in dataset_name.lower() or "chilled/aqua" in dataset_name.lower()
    is_multi_arith = "multiarith" in dataset_name.lower()
    is_last_letter = "lastletterconcat" in dataset_name.lower() or "last-letter-concatenation" in dataset_name.lower()
    is_strategyqa = "strategyqa" in dataset_name.lower()
    is_bbh = "bbh" in dataset_name.lower()  # 추가

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
        elif is_aqua:
            run_experiment_for_method_aqua(
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
        elif is_multi_arith:
            run_experiment_for_method_multiarith(
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
                with_cot_init=with_cot_init,
                token_pool=token_pool,
                random_seed=random_seed
            )
        elif is_last_letter:
            run_experiment_for_method_lastletterconcat(
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
        elif is_strategyqa:
            run_experiment_for_method_strategyqa(
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
        elif is_bbh:
            run_experiment_for_bbh(
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
            # 기본 GSM8K 형식 (fallback)
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

        # txt -> csv 변환
        output_csv_path = f"{output_dir}/{model_name}_{method_name}.csv"
        parse_txt_to_csv(output_txt_path, output_csv_path)

        # OpenAI GPT 평가
        if llm is not None:
            df = pd.read_csv(output_csv_path)
            df_eval = add_correctness_column_with_gold(df, llm=llm)
            final_csv_path = f"{output_dir}/{model_name}_{method_name}_evaluated.csv"
            df_eval.to_csv(final_csv_path, index=False, encoding='utf-8')
            print(f"Final Evaluation CSV saved -> {final_csv_path}")
        else:
            print("llm(None) -> GPT 평가 스킵")

    print("\n[모든 실험 완료]")
