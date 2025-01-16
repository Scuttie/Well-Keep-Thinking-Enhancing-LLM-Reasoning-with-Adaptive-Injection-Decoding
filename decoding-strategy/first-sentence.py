import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

def configure_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        # "meta-llama/Llama-3.1-8B-Instruct"
        "meta-llama/Llama-3.1-8B"
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


def generate_first_sentences(model, tokenizer, prompt, num_branches=10, max_length=50):
    """
    Generate the first sentence using Chain-of-Thought (CoT) decoding for branching.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    _, topk_indices = get_topk_tokens(model, inputs, num_branches)

    first_sentences = []
    response_probs = []

    for k in tqdm(range(num_branches)):
        new_input_ids = inputs.copy()
        new_input_ids['input_ids'] = torch.cat(
            [inputs['input_ids'], topk_indices[:, k].unsqueeze(-1)], dim=1
        )

        response = []
        probs = []
        
        response.append(topk_indices[:, k].item())

        for _ in range(max_length):
            topk_values, topk_indic = get_topk_tokens(model, new_input_ids, num_branches=2)
            prob_diff = topk_values[:, 0] - topk_values[:, 1]
            probs.append(prob_diff.item())
            response.append(topk_indic[:, 0].item())

            if topk_indic[:, 0] == tokenizer.eos_token_id:
                break

            new_input_ids['input_ids'] = torch.cat([new_input_ids['input_ids'], topk_indic[:, 0].unsqueeze(-1)], dim=1)

        decoded_output = tokenizer.decode(response, skip_special_tokens=True)

        first_sentence = re.split(r'(?<=[.!?])\s', decoded_output)[0]
        if first_sentence and re.match(r'^\d+\.', first_sentence):
            first_sentence = decoded_output.split(' ', 1)[1]

        first_sentences.append(first_sentence.strip())
        response_probs.append(sum(probs) / len(probs))  

    return first_sentences, response_probs


def generate_full_responses(model, tokenizer, prompt, first_sentences, max_length=500):
    """
    Generate full responses by appending each first sentence to the prompt.
    """
    device = next(model.parameters()).device
    full_responses = []

    for sentence in first_sentences:
        extended_prompt = prompt + " " + sentence
        inputs = tokenizer(extended_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        full_responses.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return full_responses

# def generate_full_responses_eos(model, tokenizer, prompt, first_sentences, max_length=500, num_branches=2):
#     """
#     Generate full responses by appending each first sentence to the prompt.
#     Handles EOS token by replacing it with the second highest probability token and continues generation.
#     """
#     device = next(model.parameters()).device
#     full_responses = []

#     for sentence in first_sentences:
#         extended_prompt = prompt + " " + sentence
#         inputs = tokenizer(extended_prompt, return_tensors="pt").to(device)

#         response = []
#         response_probs = []
#         eos_generated = False
#         # prompt_length = inputs['input_ids'].size(1)

#         while len(response) < max_length:
#             topk_values, topk_indices = get_topk_tokens(model, inputs, num_branches=num_branches)
#             # prob_diff = topk_values[:, 0] - topk_values[:, 1]
#             # response_probs.append(prob_diff.item())
#             # response.append(topk_indices[:, 0])

#             if topk_indices[:, 0] == tokenizer.eos_token_id:
#                 eos_generated = True
#                 next_token = topk_indices[:, 1]
#             else:
#                 next_token = topk_indices[:, 0]

#             inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token.unsqueeze(-1)], dim=1)

#             if len(response) >= max_length:
#                 break

#         final_response = tokenizer.decode(inputs['input_ids'][0, len(prompt):], skip_special_tokens=True)
#         full_responses.append(final_response)

#     return full_responses

def main():
    """
    Main function to generate and process responses.
    """
    model, tokenizer = configure_model()
    prompt = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
    # prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    print("Generating first sentences")
    first_sentences, response_probs = generate_first_sentences(model, tokenizer, prompt, num_branches=25)
    print("First Sentences:")
    for i, (sentence, prob) in enumerate(zip(first_sentences, response_probs)):
        print(f"{i+1}: ({prob:.4f}) {sentence}")

    print("\nGenerating full responses")
    full_responses = generate_full_responses(model, tokenizer, prompt, first_sentences)
    for i, response in enumerate(full_responses):
        print(f"\nFull Response {i+1}:\n{response}")


if __name__ == "__main__":
    main()