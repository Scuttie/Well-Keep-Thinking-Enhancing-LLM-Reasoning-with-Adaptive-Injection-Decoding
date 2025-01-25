import argparse
import os
import openai
from datasets import load_dataset
import pandas as pd

# model_utils
from model_utils.model_config import configure_model
# runner
from runner.experiment_runner import run_full_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="openai", help="LLM 타입 (ex: openai, ...)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", help="데이터셋 경로/이름")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", help="로컬 HF 모델 이름")
    parser.add_argument("--methods", nargs="+", default=["zs_cot", "step_injection", "top_k_injection", "zs_next_step"],
                        help="실험할 decoding method들 (복수 선택 가능)")
    parser.add_argument("--max_samples", type=int, default=100, help="상위 몇 개 샘플에 대해 실험할지")
    parser.add_argument("--max_length", type=int, default=500, help="디코딩 시 최대 토큰 개수")
    parser.add_argument("--output_dir", type=str, default=".", help="결과물을 저장할 디렉터리")
    args = parser.parse_args()

    # 1) OpenAI API 키 설정 (openai_api_key.txt가 있다고 가정)
    #    필요시 경로 조정
    if args.llm == "openai":
        with open("openai_api_key.txt", "r") as f:
            openai_api_key = f.read().strip()
        openai.api_key = openai_api_key
        llm = openai  # 그냥 openai 모듈 객체 자체를 넘겨서 사용
    else:
        # 만약 다른 LLM 방식을 지원하려면 여기 추가
        llm = None

    # 2) Dataset 로드
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
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
