import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def configure_model():
    """
    Configure the model and tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3").to(device)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
    return model, tokenizer


def generate_response_with_eos_rank(model, tokenizer, inputs, max_length=500):
    """
    Generate a response and replace EOS token with "stop and answer the question" once if its rank is ≤ 10.
    """
    device = next(model.parameters()).device
    inputs['input_ids'] = inputs['input_ids'].to(device)
    response = []

    prompt_length = inputs['input_ids'].size(1)  
    eos_replaced = False 

    step_token_ids = tokenizer("Step", add_special_tokens=False)["input_ids"]

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

        probabilities = torch.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

        eos_token_id = tokenizer.eos_token_id
        eos_rank = (sorted_indices == eos_token_id).nonzero(as_tuple=True)[1].item() + 1

        if not eos_replaced and eos_rank <= 10:  
            print(f"EOS Rank: {eos_rank}, EOS Probability: {probabilities[0, eos_token_id].item():.6f}")
            print("Replacing EOS.")
            for token_id in step_token_ids:
                response.append(token_id)
                next_token_tensor = torch.tensor([[token_id]], device=device)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)
            eos_replaced = True
            continue  

        next_token_id = sorted_indices[0, 0].item()

        if next_token_id == eos_token_id:
            print("EOS token encountered. Stopping generation.")
            break

        response.append(next_token_id)
        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)

    final_response = tokenizer.decode(inputs['input_ids'][0, prompt_length:], skip_special_tokens=True)
    print("\nFinal Generated Text:", final_response)
    return final_response

def run_example(prompt, model, tokenizer, max_length=500):
    """
    Run greedy decoding for a given prompt and print the results.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    final = generate_response_with_eos_rank(model, tokenizer, inputs)
    return final

# def main():
#     """
#     Main function to test greedy decoding and single EOS replacement with 'step'.
#     """
#     model, tokenizer = configure_model()
#     # prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
#     # prompt = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
#     # prompt = "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
#     prompt = "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"

#     inputs = tokenizer(prompt, return_tensors="pt")
#     generate_response_with_eos_rank(model, tokenizer, inputs)

def main():
    """
    Main function to test greedy decoding on various prompts.
    """
    model, tokenizer = configure_model()
    ## conjunction pool
    addition = [
        "and", "so", "therefore", "then", 
        "thus", "or", "in addition", "furthermore"
    ]
    contrast = [
        "however", "but", "on the other hand", "yet", 
        "in contrast", "nevertheless", "unlike", "instead", "conversely"
    ]
    mix = [
        "and", "so", "therefore", "then", 
        "thus", "or", "in addition", "furthermore",
        "however", "but", "on the other hand", "yet", 
        "in contrast", "nevertheless", "unlike", "instead", "conversely"
    ]
    reason = [
        "because", " as a result", "since", " due to this", 
        "consequently", " for this reason", " this is because"
    ]
    # example = [
    #     "for example", "for instance", "such as", 
    #     "specifically", "to illustrate", "as shown by"
    # ]
    emphasis = [
        "indeed", "in fact", "as a matter of fact", 
        "most importantly", "to clarify", "to elaborate", 
        "in other words", "that is to say"
    ]
    summary = [
        "in summary", "to wrap up", "to put it simply", 
        "all in all", "in conclusion", "lastly", "to reiterate"
    ]

    ## step pool
    # step_pool = [
    #     "Step 1", "Stage 1", "Level 1", "Phase 1", "Round 1",
    # ]
    step1 = [
        "Step 1"
    ]

    step_ = [
        "Step"
    ]

    ## article pool
    article_pool = ["The", "A", "An"]

    ## demonstratives
    demonstrative_pool = ["This", "That", "These", "Those"]
    
    nxt = ["Next Step"]

    dataset = load_dataset("openai/gsm8k", "main")
    path = "/home/hyun/eos/txt/eos-topk-mistral.txt"

    with open(path, "w", encoding="utf-8") as f:
    # 2) 테스트 데이터셋 순회
        for i, question_text in enumerate(dataset['test']['question']):
            if i >= 100:  # 100개 이상일 경우 반복 종료
                break
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

            final = run_example(question_text, model, tokenizer)

            f.write("\n--- [A] final answer ---\n")
            f.write(final + "\n")

            f.write("============================\n\n")


if __name__ == "__main__":
    main()