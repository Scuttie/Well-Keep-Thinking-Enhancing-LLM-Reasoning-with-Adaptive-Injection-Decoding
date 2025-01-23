import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from datasets import load_dataset

def configure_model(model_path):
    """
    Configure the model and load it.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate_response(prompt, model, tokenizer, max_length=500):
    """
    Generate a response from the model using greedy decoding.
    Handles conjunctions correctly, even multi-word ones.
    """

    # return final_text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    prompt_length = inputs.input_ids.size(1)

    # Generate the model's response
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=200)
    
    # Decode the output
    # Decode the final response
    final_response = tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True)
    print("\nFinal Generated Text:", final_response)
    return final_response


def main():
    """
    Main function to test greedy decoding on various prompts.
    """
    
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
    step_1 = [
        "Step 1"
    ]
    step = [
        "step"
    ]

    ## article pool
    article_pool = ["The", "A", "An"]

    ## demonstratives
    demonstrative_pool = ["This", "That", "These", "Those"]
    
    # prompts = [
    #     # """Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    #     "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
    #     # "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
    # ]
    # for prompt in prompts:
    #     run_example(prompt, model, tokenizer, addition)

    ###### 설정
    dataset_path = "openai/gsm8k"
    model_path = "lmsys/vicuna-7b-v1.5" # mistralai/Mistral-7B-v0.3
    method_explain = "greedy" #띄어쓰기 없이 #sole-step


    ###### 실행
    file_path = f"./results/{model_path.split('/')[-1]}/{dataset_path.split('/')[-1]}/{method_explain}.txt"
    dataset = load_dataset(dataset_path, "main")
    model, tokenizer = configure_model(model_path)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"\n\n============================\n")
        f.write(f"model: {model_path}\n")
        f.write(f"data: {dataset_path}\n")
        f.write(f"method: {method_explain}\n")
        f.write(f"\n============================\n\n")

        
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
            # gold_answer = dataset['test'][''][i]
# 
            f.write("\n[모범답안]\n")
            f.write(chain_of_thought + "\n")
            f.write("\n[정답]\n")
            f.write(f"#### {gold_answer}\n")

            final = generate_response(question_text, model, tokenizer,  max_length=500)

            f.write("\n--- [A] final answer ---\n")
            f.write(final + "\n")

            f.write("============================\n\n")



if __name__ == "__main__":
    main()