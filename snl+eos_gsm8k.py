import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# -------------------------------------------------
# [A] 문장 종료 판별 함수
# -------------------------------------------------
def check_sentence_end(decoded_text_so_far: str):
    """
    문장 끝 판별:
      - 마침표('.', '?', '!') 확인
      - 소수점 예외(숫자+'.')는 문장 종료로 안 봄
    """
    decoded_text_so_far = decoded_text_so_far.strip()
    if len(decoded_text_so_far) == 0:
        return False

    last_char = decoded_text_so_far[-1]
    if last_char == '.':
        # 직전 문자가 숫자이면 소수점 가능성
        if len(decoded_text_so_far) >= 2 and decoded_text_so_far[-2].isdigit():
            return False
        else:
            return True
    elif last_char in ['?', '!']:
        return True
    return False


# -------------------------------------------------
# [B] (top-k & eos 배제) 샘플링 함수
# -------------------------------------------------
def sample_single_sentence_eos_excluded(
    model,
    tokenizer,
    context_ids,
    z_mode="lowest_z",    # "lowest_z" or "highest_z"
    top_k=50,
    temperature=1.0,
    max_tokens_per_sentence=50,
):
    """
    (문장 단위) 한 번의 샘플링 (top-k 사용):
      - 문장이 끝날 때까지 토큰 생성 ('.', '?', '!' 또는 max_tokens_per_sentence 한도)
      - 각 토큰의 확률 p_i 수집 → 표준화 Z-score 계산
      - z_mode에 따라 score = min(z_i) or max(z_i)
      - eos_token 배제
    """
    device_for_model = next(model.parameters()).device
    new_ids = context_ids.clone().to(device_for_model)

    token_probs = []

    for step in range(max_tokens_per_sentence):
        outputs = model(input_ids=new_ids)
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]

        # 1) temperature 적용
        scaled_logits = last_token_logits / temperature

        # 2) top-k filtering
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        if top_k < sorted_logits.size(-1):
            sorted_logits[0, top_k:] = float('-inf')
        re_sorted_logits = torch.full_like(scaled_logits, float('-inf'))
        re_sorted_logits[0, sorted_indices] = sorted_logits[0]

        # 3) eos_token 배제
        if tokenizer.eos_token_id is not None:
            re_sorted_logits[0, tokenizer.eos_token_id] = float('-inf')

        # 4) 확률로 변환 후 샘플링
        probs = torch.softmax(re_sorted_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # 5) 이번에 뽑힌 토큰 확률 저장
        p_token = probs[0, next_token_id].item()
        token_probs.append(p_token)

        # 6) context_ids 업데이트
        new_ids = torch.cat([new_ids, next_token_id], dim=1)

        # 7) 문장 종료 검사
        decoded_so_far = tokenizer.decode(new_ids[0], skip_special_tokens=True)
        if check_sentence_end(decoded_so_far):
            break

    # 이번 문장에서 새로 생성된 부분
    added_tokens = new_ids[0, context_ids.shape[1]:]
    generated_sentence = tokenizer.decode(added_tokens, skip_special_tokens=True)

    # 점수(문장 Z-score) 계산
    if len(token_probs) > 0:
        arr_p = np.array(token_probs, dtype=np.float32)
        mean_p = arr_p.mean()
        std_p = arr_p.std() if arr_p.std() >= 1e-12 else 1e-12

        z_list = [(pval - mean_p) / std_p for pval in arr_p]
        if z_mode == "lowest_z":
            sentence_score = min(z_list)
        else:  # "highest_z"
            sentence_score = max(z_list)
    else:
        sentence_score = 0.0

    return generated_sentence, sentence_score, new_ids


# -------------------------------------------------
# [C] 문장 후보 중 best pick 
# -------------------------------------------------
def sample_sentence_candidates_and_pick_best_eos_excluded(
    model,
    tokenizer,
    context_ids,
    n_candidates=3,
    z_mode="lowest_z",   # "lowest_z" or "highest_z"
    pick_mode="max",     # "max" or "min"
    top_k=50,
    temperature=1.0,
    max_tokens_per_sentence=50
):
    """
    - n_candidates개 문장을 각각 생성 (top-k)
    - 문장 Z-score 계산 후 pick_mode='max'/'min'에 따라 best 선택
    """
    candidates = []
    for _ in range(n_candidates):
        gen_sentence, score, new_ids = sample_single_sentence_eos_excluded(
            model=model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            z_mode=z_mode,
            top_k=top_k,
            temperature=temperature,
            max_tokens_per_sentence=max_tokens_per_sentence
        )
        candidates.append((gen_sentence, score, new_ids))

    if pick_mode == "max":
        best_idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
    else:
        best_idx = min(range(len(candidates)), key=lambda i: candidates[i][1])

    best_sentence, best_score, best_ids = candidates[best_idx]
    return best_sentence, best_score, best_ids


# -------------------------------------------------
# [D] 문장 단위 생성 (top-k 버전)
#     (원 코드: generate_text_sentence_by_sentence_eos_excluded)
# -------------------------------------------------
def generate_text_sentence_by_sentence_eos_excluded(
    model,
    tokenizer,
    prompt,
    max_sentences=5,
    n_candidates=3,
    z_mode="lowest_z",   # "lowest_z" or "highest_z"
    pick_mode="max",     # "max" or "min"
    top_k=50,
    temperature=1.0,
    max_tokens_per_sentence=50
):
    """
    문장 단위로 top-k 생성
    """
    device_for_model = next(model.parameters()).device
    context_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_for_model)

    final_text = tokenizer.decode(context_ids[0], skip_special_tokens=True)

    for _ in range(max_sentences):
        best_sentence, best_score, best_ids = sample_sentence_candidates_and_pick_best_eos_excluded(
            model=model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            n_candidates=n_candidates,
            z_mode=z_mode,
            pick_mode=pick_mode,
            top_k=top_k,
            temperature=temperature,
            max_tokens_per_sentence=max_tokens_per_sentence
        )
        context_ids = best_ids
        final_text += best_sentence

    return final_text


# -------------------------------------------------
# [E] 모델 로드 함수 예시 (실제 환경에 맞추어 수정)
# -------------------------------------------------
def load_llama7b_model(model_path, device="cuda"):
    """
    실제 환경/모델에 맞춰 수정해 주세요.
    (예시: meta-llama/Llama-3.1-8B 로드)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


# -------------------------------------------------
# [F] GSM8K 로드 & 실험 (top-k 버전)
# -------------------------------------------------
if __name__ == "__main__":

    # (1) 모델 로드 
    # model_path = "meta-llama/Llama-3.1-8B"
    # tokenizer, model = load_llama7b_model(model_path, device="cuda")

    # (2) GSM8K 데이터 로드
    #    pip install datasets 후, 아래 코드를 이용
    dataset = load_dataset("openai/gsm8k", "main")

    # (3) 결과 저장할 txt 파일 경로
    save_path = "txt파일경로.txt"

    # (4) 파일 열기 (덮어쓰기 'w' or 이어쓰기 'a')
    with open(save_path, "w", encoding="utf-8") as f:
        # (5) GSM8K의 test split 순회
        for i, question_text in enumerate(dataset['test']['question']):
            f.write(f"\n============================\n")
            f.write(f"Test sample #{i+1}\n")
            f.write(f"Question: {question_text}\n")

            # ---- 실제 모범답안/정답 표시 ----
            cot_with_answer = dataset['test']['answer'][i]
            splitted = cot_with_answer.split("####")
            chain_of_thought = splitted[0].strip() if len(splitted) > 0 else ""
            gold_answer = splitted[1].strip() if len(splitted) > 1 else ""

            f.write("\n[모범답안]\n")
            f.write(chain_of_thought + "\n")
            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            # ---- 4가지 Standardized Mode (top-k)로 결과 생성 ----
            max_sentences = 10
            n_candidates = 5
            top_k = 50
            temperature = 1.0
            max_tokens_per_sentence = 50

            # (A) lowest_z + max
            ans_a = generate_text_sentence_by_sentence_eos_excluded(
                model=model,
                tokenizer=tokenizer,
                prompt=question_text,
                max_sentences=max_sentences,
                n_candidates=n_candidates,
                z_mode="lowest_z",
                pick_mode="max",
                top_k=top_k,
                temperature=temperature,
                max_tokens_per_sentence=max_tokens_per_sentence
            )

            # (B) lowest_z + min
            ans_b = generate_text_sentence_by_sentence_eos_excluded(
                model=model,
                tokenizer=tokenizer,
                prompt=question_text,
                max_sentences=max_sentences,
                n_candidates=n_candidates,
                z_mode="lowest_z",
                pick_mode="min",
                top_k=top_k,
                temperature=temperature,
                max_tokens_per_sentence=max_tokens_per_sentence
            )

            # (C) highest_z + max
            ans_c = generate_text_sentence_by_sentence_eos_excluded(
                model=model,
                tokenizer=tokenizer,
                prompt=question_text,
                max_sentences=max_sentences,
                n_candidates=n_candidates,
                z_mode="highest_z",
                pick_mode="max",
                top_k=top_k,
                temperature=temperature,
                max_tokens_per_sentence=max_tokens_per_sentence
            )

            # (D) highest_z + min
            ans_d = generate_text_sentence_by_sentence_eos_excluded(
                model=model,
                tokenizer=tokenizer,
                prompt=question_text,
                max_sentences=max_sentences,
                n_candidates=n_candidates,
                z_mode="highest_z",
                pick_mode="min",
                top_k=top_k,
                temperature=temperature,
                max_tokens_per_sentence=max_tokens_per_sentence
            )

            f.write("\n--- [A] lowest_z + max ---\n")
            f.write(ans_a + "\n")
            f.write("\n--- [B] lowest_z + min ---\n")
            f.write(ans_b + "\n")
            f.write("\n--- [C] highest_z + max ---\n")
            f.write(ans_c + "\n")
            f.write("\n--- [D] highest_z + min ---\n")
            f.write(ans_d + "\n")

            f.write("============================\n\n")

    print(f"모든 결과가 다음 경로에 저장되었습니다: {save_path}")
