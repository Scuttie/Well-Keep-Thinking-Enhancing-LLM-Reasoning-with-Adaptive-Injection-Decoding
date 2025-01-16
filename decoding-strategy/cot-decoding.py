import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def configure_model():
    """
    Configure the model to load Mistral-7B-Instruct.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B").to(device)
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


def filter_whitespace_tokens(topk_values, topk_indices, tokenizer, num_branches):
    """
    Filter out whitespace tokens from the top-k tokens.
    """
    filtered_indices = []
    filtered_values = []

    for i in range(topk_indices.size(1)):
        token = tokenizer.decode([topk_indices[0, i].item()])
        if not token.isspace():  
            filtered_indices.append(topk_indices[0, i])
            filtered_values.append(topk_values[0, i])
            if len(filtered_indices) >= num_branches:
                break

    indices = torch.tensor([filtered_indices], device=topk_indices.device)
    values = torch.tensor([filtered_values], device=topk_values.device)

    return values, indices


def generate_response(model, tokenizer, inputs, max_length=500):
    """
    Generate a response from the model using CoT decoding.
    """
    response = []
    response_probs = []

    device = next(model.parameters()).device
    inputs['input_ids'] = inputs['input_ids'].to(device)

    for _ in range(max_length):
        topk_values, topk_indices = get_topk_tokens(model, inputs, num_branches=2)
        prob_diff = topk_values[:, 0] - topk_values[:, 1]
        response_probs.append(prob_diff.item())
        response.append(topk_indices[:, 0])

        if topk_indices[:, 0] == tokenizer.eos_token_id:
            break

        inputs['input_ids'] = torch.cat([inputs['input_ids'], topk_indices[:, 0].unsqueeze(-1)], dim=1)

    return inputs['input_ids'], response_probs


def generate_branching_responses(model, tokenizer, prompt, num_branches=10, max_length=500):
    """
    Generate branching responses using CoT decoding.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    topk_values, topk_indices = get_topk_tokens(model, inputs, num_branches + 5)  
    topk_values, topk_indices = filter_whitespace_tokens(topk_values, topk_indices, tokenizer, num_branches)

    filtered_branches = topk_values.size(1)
    print("\nTop-k Tokens and Probabilities:")
    for i in range(filtered_branches):
        token = tokenizer.decode(topk_indices[0, i].item())
        probability = topk_values[0, i].item()
        print(f"{i+1}: Token: '{token}' | Probability: {probability:.6f}")

    responses = []
    response_probs = []

    for k in tqdm(range(filtered_branches), desc="Generating branches"):
        new_input_ids = inputs.copy()
        new_input_ids['input_ids'] = torch.cat(
            [inputs['input_ids'], topk_indices[:, k].unsqueeze(-1)], dim=1
        )

        response, probs = generate_response(model, tokenizer, new_input_ids, max_length)
        responses.append(tokenizer.batch_decode(response))
        response_probs.append(sum(probs) / len(probs))

    return responses, response_probs


def run_example(prompt, model, tokenizer, num_branches=10, max_length=500):
    """
    Run CoT decoding for a given prompt and print the results.
    """
    responses, response_probs = generate_branching_responses(model, tokenizer, prompt, num_branches, max_length)

    print('Prompt:', prompt)
    for k, (response, prob) in enumerate(zip(responses, response_probs)):
        print(f'\nResponse k={k}:\n\n', response[0][len(prompt)+1:])
        print('\nScore:', prob)


def main():
    """
    Main function to test CoT decoding on various prompts.
    """
    model, tokenizer = configure_model()

    prompts = [
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
        # "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"
    ]
    for prompt in prompts:
        run_example(prompt, model, tokenizer)


if __name__ == "__main__":
    main()