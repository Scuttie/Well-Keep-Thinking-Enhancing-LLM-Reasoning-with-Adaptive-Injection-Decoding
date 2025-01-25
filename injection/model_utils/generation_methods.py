import torch

def generate_greedy_decoding(model, tokenizer, prompt, max_length=300):
    """
    단순 Greedy Decoding
    EOS 토큰이 나오면 종료.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    response_ids = []

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # EOS 검사
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # 토큰 추가
        response_ids.append(next_token_id.item())
        next_token_tensor = torch.tensor([[next_token_id.item()]], device=device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)

    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    return decoded


def generate_zero_shot_cot(model, tokenizer, prompt, max_length=300):
    """
    Zero-Shot CoT:
    질문 뒤에 "Let's think step by step."를 붙여서 Greedy 디코딩.
    """
    cot_prompt = prompt.strip() + "\nLet's think step by step."
    return generate_greedy_decoding(model, tokenizer, cot_prompt, max_length=max_length)


def generate_step_injection(model, tokenizer, prompt, max_length=300):
    """
    Step-Injection:
    Greedy Decoding 도중 EOS 토큰이 나오면 'Step'으로 대체하고 계속 디코딩.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    response_ids = []
    step_token_ids = tokenizer.encode("Step", add_special_tokens=False)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # EOS 검사
        if next_token_id.item() == tokenizer.eos_token_id:
            # EOS 시점에 "Step" 삽입
            for s_id in step_token_ids:
                response_ids.append(s_id)
                next_token_tensor = torch.tensor([[s_id]], device=device)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)
        else:
            response_ids.append(next_token_id.item())
            next_token_tensor = torch.tensor([[next_token_id.item()]], device=device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)

    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    return decoded


def generate_top_k_injection(model, tokenizer, prompt, k=10, max_length=300):
    """
    Top-K Injection:
    디코딩 중 EOS 토큰이 Top-K 범위 내에 존재하면 (아직 한 번도 치환 안 했다면) 'Step'을 삽입.
    이후에는 일반 greedy로 진행하며, 실제 EOS가 나오면 중단.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    response_ids = []
    eos_replaced = False

    step_token_ids = tokenizer.encode("Step", add_special_tokens=False)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        # 확률 계산
        probs = torch.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # eos 토큰의 rank 계산
        eos_rank_positions = (sorted_indices[0] == eos_id).nonzero(as_tuple=True)
        if len(eos_rank_positions[0]) > 0:
            eos_rank_idx = eos_rank_positions[0].item()
        else:
            eos_rank_idx = -1  # eos가 top-k 안에 없음을 의미

        # 아직 치환 안 했고 eos가 top-k 안에 있으면 치환
        if (not eos_replaced) and (0 <= eos_rank_idx < k):
            # eos 토큰이 top-k 범위 내 -> "Step" 삽입
            for s_id in step_token_ids:
                response_ids.append(s_id)
                next_token_tensor = torch.tensor([[s_id]], device=device)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)
            eos_replaced = True
            continue

        # 그 외에는 greedy top-1 선택
        next_token_id = sorted_indices[0, 0].item()
        if next_token_id == eos_id:
            # 실제 EOS 토큰 -> 중단
            break

        response_ids.append(next_token_id)
        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)

    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    return decoded


def generate_zs_next_step(model, tokenizer, prompt, max_length=300):
    """
    ZS + "Next Step":
    - 초기 프롬프트로 "Let's think step by step." 붙여서 시작
    - 디코딩 중, top-1이 EOS 토큰이면 "Next step" 삽입
    """
    device = next(model.parameters()).device
    cot_prompt = prompt.strip() + "\nLet's think step by step."
    inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)

    response_ids = []
    next_step_token_ids = tokenizer.encode("Next step", add_special_tokens=False)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # EOS 검사
        if next_token_id.item() == tokenizer.eos_token_id:
            # EOS 시점에 "Next step" 삽입
            for ns_id in next_step_token_ids:
                response_ids.append(ns_id)
                next_token_tensor = torch.tensor([[ns_id]], device=device)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)
        else:
            response_ids.append(next_token_id.item())
            next_token_tensor = torch.tensor([[next_token_id.item()]], device=device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)

    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    return decoded
