import argparse
import os
import openai
import json
from datasets import Dataset
from datasets import load_dataset
import pandas as pd

# model_utils
from model_utils.model_config import configure_model
# runner
from runner.experiment_runner import run_full_pipeline

def load_commonsenseqa():
    """
    CommonsenseQA용 JSONL 로더
    """
    file_path = "/home/jewonyeom/LLMs-Are-Just-Shy_Reasoning-by-Intervention/injection/train_rand_split.jsonl"
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            stem = obj['question']['stem']
            answer = obj['answerKey']
            # 선택지 합치기
            choices = "\n".join([f"({choice['label']}) {choice['text']}"
                                 for choice in obj['question']['choices']])
            question = f"{stem}\n{choices}"
            data.append({"question": question, "answer": answer})

    return Dataset.from_list(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="openai",
                        help="LLM 타입 (ex: openai, none ...)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500",
                        help="데이터셋 경로/이름 (예: 'CommonsenseQA' or 'openai/gsm8k' 등)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="로컬 HF 모델 이름")
    parser.add_argument("--methods", nargs="+",
                        default=["zs_cot", "step_injection", "top_k_injection", "zs_next_step"],
                        help="실험할 decoding method들 (복수 선택 가능)")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="상위 몇 개 샘플에 대해 실험할지")
    parser.add_argument("--max_length", type=int, default=500,
                        help="디코딩 시 최대 토큰 개수")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="결과물을 저장할 디렉터리")
    parser.add_argument("--top_k_value", type=int, default=10,
                        help="(top_k_injection)에서 사용할 k 값 (기본=10)")
    parser.add_argument("--injection_token", type=str, default="Step",
                        help="(top_k_injection, zs_next_step 등)에서 사용할 삽입 문자열 (기본='Step')")
    parser.add_argument("--patience_value", type=int, default=50,
                        help="(zs_top_k_multi_inject_patience)에서 사용할 patience 길이 (기본=50)")
    parser.add_argument("--with_cot_init", action="store_true",
                        help="top_k_injection 시, prompt 앞에 'Let's think step by step.'를 넣을지 여부")


    args = parser.parse_args()

    # 1) OpenAI API 키 설정 (llm == openai면)
    if args.llm == "openai":
        with open("openai_api_key.txt", "r") as f:
            openai_api_key = f.read().strip()
        openai.api_key = openai_api_key
        llm = openai
    else:
        llm = None

    # 2) Dataset 로드 분기
    if args.dataset.lower() == "commonsenseqa":
        dataset = load_commonsenseqa()
    elif "math-500" in args.dataset.lower():
        # MATH-500
        dataset = load_dataset(args.dataset)
    else:
        # GSM8K 등
        dataset = load_dataset(args.dataset)

    # 3) 로컬 모델 & 토크나이저 로드
    model, tokenizer = configure_model(args.model_name)

    # 모델 짧은 이름
    model_short = args.model_name.split("/")[-1]

    # 4) 전체 파이프라인 실행
    run_full_pipeline(
        llm=llm,
        model=model,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        dataset=dataset,
        model_name=model_short,
        methods=args.methods,
        max_samples=args.max_samples,
        max_length=args.max_length,
        output_dir=args.output_dir,
        top_k_value=args.top_k_value,
        injection_token=args.injection_token,
        patience_value=args.patience_value,
        with_cot_init=args.with_cot_init
    )

if __name__ == "__main__":
    main()
