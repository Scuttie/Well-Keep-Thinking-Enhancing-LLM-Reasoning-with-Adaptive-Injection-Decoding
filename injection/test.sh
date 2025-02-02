#!/bin/bash
#SBATCH --job-name=commonqa_mistral_none
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --partition=laal_a6000

# 추가: 실험할 method
METHODS=("top_k_injection")
# 실험할 모델 #"meta-llama/Llama-3.1-8B" #"mistralai/Mistral-7B-v0.3"
MODELS=("mistralai/Mistral-7B-v0.3")
# injection token 후보
TOKENS=("step" "Next step" "Well" "Okay")

# top_k_injection에서 시험할 k값들
K_VALUES=(10)
# with_cot_init on/off
WITH_COT_INIT_OPTIONS=("false")

# 출력 디렉토리
OUTPUT_DIR="./results_commonsenseqa"

# 최대 샘플 수
MAX_SAMPLES=100
# 디코딩 최대 토큰
MAX_LENGTH=500

for model_name in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    for token in "${TOKENS[@]}"; do

      # 만약 method가 top_k_injection이면 k와 with_cot_init도 바꿔가며 실행
      if [ "$method" == "top_k_injection" ]; then
        for k_val in "${K_VALUES[@]}"; do
          for init_opt in "${WITH_COT_INIT_OPTIONS[@]}"; do
            echo "====================================================="
            echo "Run with model=$model_name, method=$method, token=$token, k=$k_val, with_cot_init=$init_opt"

            python main.py \
              --llm openai \
              --dataset "CommonsenseQA" \
              --model_name "$model_name" \
              --methods "$method" \
              --injection_token "$token" \
              --max_samples $MAX_SAMPLES \
              --max_length $MAX_LENGTH \
              --top_k_value $k_val \
              --output_dir "$OUTPUT_DIR/${model_name//\//-}_${method}_${token}_k${k_val}_init${init_opt}" \
              $( [ "$init_opt" == "true" ] && echo "--with_cot_init" )
          done
        done
      else
        # 그 외 방법(zs_next_step 등)은 그냥 기존처럼
        echo "====================================================="
        echo "Run with model=$model_name, method=$method, token=$token"
        python main.py \
          --llm openai \
          --dataset "CommonsenseQA" \
          --model_name "$model_name" \
          --methods "$method" \
          --injection_token "$token" \
          --max_samples $MAX_SAMPLES \
          --max_length $MAX_LENGTH \
          --output_dir "$OUTPUT_DIR/${model_name//\//-}_${method}_${token}"
      fi
    done
  done
done
