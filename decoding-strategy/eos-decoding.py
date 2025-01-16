import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm


def configure_model():
    """
    Configure the model to use 4-bit quantization and load model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        # "meta-llama/Llama-3.1-8B-Instruct", 
        "meta-llama/Llama-3.1-8B", 
        # quantization_config=config
    ).to(device)
    
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    return model, tokenizer

def get_topk_tokens(model, inputs, num_branches=10):
    """
    Get the top-k tokens and their probabilities for a given input.
    """
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

    probabilities = torch.softmax(next_token_logits, dim=-1)
    topk_values, topk_indices = torch.topk(probabilities, num_branches)

    return topk_values, topk_indices

## every step
# def generate_response(model, tokenizer, inputs, max_length=500):
#     """
#     Generate a response from the model using CoT decoding.
#     """
#     response = []
#     response_probs = []

#     device = next(model.parameters()).device
#     inputs['input_ids'] = inputs['input_ids'].to(device)

#     eos_generated = False
#     while len(response) < max_length:
#         topk_values, topk_indices = get_topk_tokens(model, inputs, num_branches=5)
#         prob_diff = topk_values[:, 0] - topk_values[:, 1]
#         response_probs.append(prob_diff.item())
#         response.append(topk_indices[:, 0])

#         # If EOS token is encountered, replace it with the second most probable token and continue
#         if topk_indices[:, 0] == tokenizer.eos_token_id:
#             print("EOS encountered, replacing with second highest probability token.")
#             inputs['input_ids'] = torch.cat([inputs['input_ids'], topk_indices[:, 1].unsqueeze(-1)], dim=1)
#         else:
#             # Continue normally if EOS is not encountered
#             inputs['input_ids'] = torch.cat([inputs['input_ids'], topk_indices[:, 0].unsqueeze(-1)], dim=1)

#         # Print the current response and top-k tokens with probabilities
#         print("Generated Text so far:", tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
#         print("Top 5 tokens and their probabilities:")
#         for i in range(topk_values.size(1)):  # Iterate over the top-k tokens
#             print(f"Token: {tokenizer.decode(topk_indices[0, i].unsqueeze(0))}, Probability: {topk_values[0, i].item()}")

#     final_response = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
#     return final_response, response_probs


def generate_response(model, tokenizer, inputs, max_length=500):
    """
    Generate a response from the model using CoT decoding.
    """
    response = []
    response_probs = []

    device = next(model.parameters()).device
    inputs['input_ids'] = inputs['input_ids'].to(device)

    eos_generated = False
    prompt_length = inputs['input_ids'].size(1)  # Get the length of the prompt

    while len(response) < max_length:
        topk_values, topk_indices = get_topk_tokens(model, inputs, num_branches=5)
        prob_diff = topk_values[:, 0] - topk_values[:, 1]
        response_probs.append(prob_diff.item())
        response.append(topk_indices[:, 0])

        if topk_indices[:, 0] == tokenizer.eos_token_id:
            print("EOS encountered, replacing with second highest probability token.")
            inputs['input_ids'] = torch.cat([inputs['input_ids'], topk_indices[:, 1].unsqueeze(-1)], dim=1)
            
            print("Generated Text so far:", tokenizer.decode(inputs['input_ids'][0, prompt_length:], skip_special_tokens=True))
            print("Top 5 tokens and their probabilities:")
            for i in range(topk_values.size(1)): 
                print(f"Token: {tokenizer.decode(topk_indices[0, i].unsqueeze(0))}, Probability: {topk_values[0, i].item()}")
        else:
            inputs['input_ids'] = torch.cat([inputs['input_ids'], topk_indices[:, 0].unsqueeze(-1)], dim=1)

        if len(response) >= max_length:
            final_response = tokenizer.decode(inputs['input_ids'][0, prompt_length:], skip_special_tokens=True)
            print("\nFinal Generated Text (max length reached):", final_response)
            break

    final_response = tokenizer.decode(inputs['input_ids'][0, prompt_length:], skip_special_tokens=True)
    return final_response, response_probs

def run_example(prompt, model, tokenizer, max_length=500):
    """
    Run CoT decoding for a given prompt and print the results.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    inputs['input_ids'] = inputs['input_ids'].to(device)
    response, response_probs = generate_response(model, tokenizer, inputs, max_length)


def main():
    """
    Main function to test CoT decoding on various prompts.
    """
    model, tokenizer = configure_model()

    prompts = [
    """Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?""",
                #  "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
                #  "[INST]Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?[/INST]",
                #  "[INST]James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?[/INST]" ]
    ]
    for prompt in prompts:
        run_example(prompt, model, tokenizer)


if __name__ == "__main__":
    main()