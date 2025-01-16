import os
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------
# [1] 모델/토크나이저 로딩
# -------------------------------------------------
def load_llama7b_model(model_path, device="cuda"):
    """
    모델과 토크나이저를 로딩하는 함수.
    device_map='auto' 사용 시 여러 GPU 또는 CPU에 자동 배치 가능.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"  # 또는 .to(device)
    )
    model.eval()
    return tokenizer, model

# -------------------------------------------------
# [2] 문장 종료 판별 함수
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
# [3] 한 문장 샘플링 & Z-score 계산
# -------------------------------------------------
def sample_single_sentence(
    model,
    tokenizer,
    context_ids,
    z_mode="lowest_z",    # "lowest_z" or "highest_z"
    top_p=0.9,
    temperature=1.0,
    max_tokens_per_sentence=50,
):
    """
    (문장 단위) 한 번의 샘플링:
      1) 문장이 끝날 때까지 토큰을 생성( '.', '?', '!' 또는 eos_token )
      2) 각 토큰 확률 수집 -> 표준화 z_i = (p_i - mean_p)/std_p
      3) z_mode에 따라 score = min(z_i) or max(z_i)
      4) 생성된 문장, 점수(sentence_score), 새 context_ids 반환
    """
    device_for_model = next(model.parameters()).device
    new_ids = context_ids.clone().to(device_for_model)

    token_probs = []
    generated_token_count = 0

    for step in range(max_tokens_per_sentence):
        outputs = model(input_ids=new_ids)
        logits = outputs.logits
        last_token_logits = logits[:, -1, :]

        # temperature 적용
        scaled_logits = last_token_logits / temperature

        # top-p filtering
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        idx_remove = (cprobs > top_p)
        if idx_remove.any():
            first_true_idx = torch.nonzero(idx_remove, as_tuple=True)[1][0].item()
            sorted_logits[0, first_true_idx+1:] = float('-inf')

        # 정렬 전 상태로 재매핑
        re_sorted_logits = torch.full_like(scaled_logits, float('-inf'))
        re_sorted_logits[0, sorted_indices] = sorted_logits[0]

        probs = torch.softmax(re_sorted_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # 다음 토큰 확률 저장
        p_token = probs[0, next_token_id].item()
        token_probs.append(p_token)
        generated_token_count += 1

        new_ids = torch.cat([new_ids, next_token_id], dim=1)

        # eos 검사
        if tokenizer.eos_token_id is not None:
            if next_token_id.item() == tokenizer.eos_token_id:
                break

        # 문장 종료 검사
        decoded_so_far = tokenizer.decode(new_ids[0], skip_special_tokens=True)
        if check_sentence_end(decoded_so_far):
            break

    # 이번 문장에서 새로 생성된 부분
    added_tokens = new_ids[0, context_ids.shape[1]:]
    generated_sentence = tokenizer.decode(added_tokens, skip_special_tokens=True)

    # 점수 계산(=문장 스코어) : z_mode에 따라 min(z_i) or max(z_i)
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
# [4] n_candidates 문장을 생성해 그중 best pick 고르기
# -------------------------------------------------
def sample_sentence_candidates_and_pick_best(
    model,
    tokenizer,
    context_ids,
    n_candidates=3,
    z_mode="lowest_z",   # "lowest_z" or "highest_z"
    pick_mode="max",     # "max" or "min"
    top_p=0.9,
    temperature=1.0,
    max_tokens_per_sentence=50
):
    """
    - n_candidates개 문장을 각각 생성
    - 각 문장 점수(sentence_score) 계산
    - pick_mode='max' 이면 score가 가장 큰 후보, 'min'이면 가장 작은 후보 선택
    """
    candidates = []
    for _ in range(n_candidates):
        gen_sentence, score, new_ids = sample_single_sentence(
            model=model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            z_mode=z_mode,
            top_p=top_p,
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
# [5] 문장 단위로 생성 (문장마다 n_candidates 중 최적 후보 고르는 방식)
# -------------------------------------------------
def generate_text_sentence_by_sentence(
    model,
    tokenizer,
    prompt,
    max_sentences=5,
    n_candidates=3,
    z_mode="lowest_z",   # "lowest_z" or "highest_z"
    pick_mode="max",     # "max" or "min"
    top_p=0.9,
    temperature=1.0,
    max_tokens_per_sentence=50
):
    """
    문장 단위로 생성하며:
      - 각 문장마다 n_candidates 개를 샘플링
      - z_mode로 점수를 구한 후 pick_mode에 따라 가장 적절한 문장 선택
      - 최종적으로 최대 max_sentences개 문장을 연결
    """
    device_for_model = next(model.parameters()).device
    context_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_for_model)

    final_text = tokenizer.decode(context_ids[0], skip_special_tokens=True)

    for _ in range(max_sentences):
        best_sentence, best_score, best_ids = sample_sentence_candidates_and_pick_best(
            model=model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            n_candidates=n_candidates,
            z_mode=z_mode,
            pick_mode=pick_mode,
            top_p=top_p,
            temperature=temperature,
            max_tokens_per_sentence=max_tokens_per_sentence
        )

        context_ids = best_ids
        final_text += best_sentence

        # eos_token 확인
        if tokenizer.eos_token_id is not None:
            if context_ids[0, -1].item() == tokenizer.eos_token_id:
                break

    return final_text

# -------------------------------------------------
# [6] (옵션) Greedy Decoding 예시
# -------------------------------------------------
def generate_text_sentence_by_sentence_greedy(
    model,
    tokenizer,
    prompt,
    max_sentences=5,
    max_tokens_per_sentence=50,
):
    """
    간단한 Greedy 방식으로 문장 단위 생성 (n_candidates=1, top_p=1.0, temperature=1.0...)
    """
    device_for_model = next(model.parameters()).device
    context_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_for_model)

    final_text = tokenizer.decode(context_ids[0], skip_special_tokens=True)

    for _ in range(max_sentences):
        with torch.no_grad():
            for _step in range(max_tokens_per_sentence):
                outputs = model(input_ids=context_ids)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]

                # Greedy: argmax
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                context_ids = torch.cat([context_ids, next_token_id], dim=1)

                if tokenizer.eos_token_id is not None:
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break

                # 문장 끝 여부
                decoded_so_far = tokenizer.decode(context_ids[0], skip_special_tokens=True)
                if check_sentence_end(decoded_so_far):
                    break

        # 이번 루프에서 새로 추가된 문장만 추출
        new_generated = tokenizer.decode(context_ids[0], skip_special_tokens=True)
        final_text = new_generated

        if tokenizer.eos_token_id is not None:
            if context_ids[0, -1].item() == tokenizer.eos_token_id:
                break

    return final_text

if __name__ == "__main__":

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main")
    
    model_path = "meta-llama/Llama-3.1-8B"
    tokenizer, model = load_llama7b_model(model_path, device="cuda")

    # 결과를 저장할 txt 파일 경로 지정 (필요에 따라 파일명 변경)
    save_path = "path.txt"

    # (예시) 기존에 있던 파일을 덮어쓰고 새로 시작하려면 'w',
    #        이어서 계속 쓰려면 'a' 모드로 열면 됩니다.
    with open(save_path, "w", encoding="utf-8") as f:
        # 2) 테스트 데이터셋 순회
        for i, question_text in enumerate(dataset['test']['question']):
            # 기존 print 대신 txt 파일에 기록
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

            # ---- 4가지 Standardized Mode로 결과 생성 ----
            max_sentences = 10
            n_candidates = 10
            top_p = 0.9
            temperature = 1.0
            max_tokens_per_sentence = 50

            # (A) lowest_z + max
            ans_a = generate_text_sentence_by_sentence(
                model=model,
                tokenizer=tokenizer,
                prompt=question_text,
                max_sentences=max_sentences,
                n_candidates=n_candidates,
                z_mode="lowest_z",
                pick_mode="max",
                top_p=top_p,
                temperature=temperature,
                max_tokens_per_sentence=max_tokens_per_sentence
            )

            # (B) lowest_z + min
            ans_b = generate_text_sentence_by_sentence(
                model=model,
                tokenizer=tokenizer,
                prompt=question_text,
                max_sentences=max_sentences,
                n_candidates=n_candidates,
                z_mode="lowest_z",
                pick_mode="min",
                top_p=top_p,
                temperature=temperature,
                max_tokens_per_sentence=max_tokens_per_sentence
            )

            # (C) highest_z + max
            ans_c = generate_text_sentence_by_sentence(
                model=model,
                tokenizer=tokenizer,
                prompt=question_text,
                max_sentences=max_sentences,
                n_candidates=n_candidates,
                z_mode="highest_z",
                pick_mode="max",
                top_p=top_p,
                temperature=temperature,
                max_tokens_per_sentence=max_tokens_per_sentence
            )

            # (D) highest_z + min
            ans_d = generate_text_sentence_by_sentence(
                model=model,
                tokenizer=tokenizer,
                prompt=question_text,
                max_sentences=max_sentences,
                n_candidates=n_candidates,
                z_mode="highest_z",
                pick_mode="min",
                top_p=top_p,
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
