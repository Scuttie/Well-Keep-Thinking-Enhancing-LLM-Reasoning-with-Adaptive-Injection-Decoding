import torch

def generate_greedy_decoding(model, tokenizer, prompt, max_length=300):
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


def generate_step_injection(model, tokenizer, prompt,
                            max_length=300,
                            injection_token="Step"):
    """
    Step-Injection:
    디코딩 도중 EOS 토큰이 나오면, 그 자리에 injection_token 삽입 후 계속 디코딩
    (무한루프 방지를 위해 max_length까지는 진행).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    response_ids = []
    step_token_ids = tokenizer.encode(injection_token, add_special_tokens=False)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # EOS 검사
        if next_token_id.item() == tokenizer.eos_token_id:
            # EOS 시점에 injection_token 삽입
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


def generate_top_k_injection(model, tokenizer, prompt,
                             k=10, injection_token="Step",
                             max_length=300,
                             with_cot_init=False 
                             ):
    """
    Top-K Injection + (option) CoT init.
    """
    if with_cot_init:
        print("with_cot_init Called.")
        prompt = prompt.strip() + "\nLet's think step by step."

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    response_ids = []
    eos_replaced = False

    injection_ids = tokenizer.encode(injection_token, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # eos 토큰이 top-k 안에 있는지 확인
        eos_rank_positions = (sorted_indices[0] == eos_id).nonzero(as_tuple=True)
        if len(eos_rank_positions[0]) > 0:
            eos_rank_idx = eos_rank_positions[0].item()
        else:
            eos_rank_idx = -1

        # 아직 치환 안 했고, eos가 top-k 안에 있으면 injection_token으로 치환
        if (not eos_replaced) and (0 <= eos_rank_idx < k):
            for t_id in injection_ids:
                response_ids.append(t_id)
                next_token_tensor = torch.tensor([[t_id]], device=device)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)
            eos_replaced = True
            continue

        # 그 외엔 top-1(greedy)
        next_token_id = sorted_indices[0, 0].item()
        if next_token_id == eos_id:
            # 실제로 EOS가 그리디 top-1 이라면 종료
            break

        response_ids.append(next_token_id)
        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)

    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    return decoded

