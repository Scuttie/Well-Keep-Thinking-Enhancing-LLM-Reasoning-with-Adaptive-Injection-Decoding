import pandas as pd
import re

def parse_text_file(input_file, output_csv):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by test sample blocks
    test_samples = content.split("============================")
    parsed_data = []

    for sample in test_samples:
        sample = sample.strip()
        if not sample:
            continue

        # Extract question
        question_match = re.search(r"Question:\s*(.+)", sample)
        question = question_match.group(1).strip() if question_match else None

        # Extract model answer (모범답안)
        reasoning_match = re.search(r"\[모범답안\]\s*(.*?)(?:\n\n|\[정답\])", sample, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        # Extract answer
        answer_match = re.search(r"\[정답\]\s*####\s*(.+)", sample)
        answer = answer_match.group(1).strip() if answer_match else None

        # Extract generated answer
        generated_answer_match = re.search(r"--- \[A\] final answer ---\n(.+)", sample, re.DOTALL)
        generated_answer = generated_answer_match.group(1).strip() if generated_answer_match else None

        if question and answer and reasoning and generated_answer:
            parsed_data.append({
                "Question": question,
                "Correct-Reasoning": reasoning,
                "Answer": answer,
                "Generated-Answer": generated_answer
            })

# def parse_text_file(input_file, output_csv):
#     with open(input_file, 'r', encoding='utf-8') as file:
#         content = file.read()
    
#     # Split by test sample blocks
#     test_samples = content.split("============================")
#     parsed_data = []

#     for sample in test_samples:
#         sample = sample.strip()
#         if not sample:
#             continue

#         # Extract question
#         question_match = re.search(r"Question:\s*(.+)", sample)
#         question = question_match.group(1).strip() if question_match else None

#         # Extract model answer (모범답안)
#         reasoning_match = re.search(r"\[모범답안\]\s*(.*?)(?:\n\n|\[정답\])", sample, re.DOTALL)
#         reasoning = reasoning_match.group(1).strip() if reasoning_match else None

#         # Extract answer
#         answer_match = re.search(r"\[정답\]\s*####\s*(.+)", sample)
#         answer = answer_match.group(1).strip() if answer_match else None

#         # Extract generated answer
#         min_generated_answer_match = re.search(r"--- [A] min uncertainty ---\n(.+)", sample, re.DOTALL)
#         min_generated_answer = min_generated_answer_match.group(1).strip() if min_generated_answer_match else None

#         if question and answer and reasoning and generated_answer:
#             parsed_data.append({
#                 "Question": question,
#                 "Correct-Reasoning": reasoning,
#                 "Answer": answer,
#                 "Generated-Answer": generated_answer
#             })

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(parsed_data)
    df.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"Parsed data saved to {output_csv}")

# Example usage
input_file = "/home/hyun/eos/sole-step-once-injection.txt" 
output_csv = "./sole-step-once-injection.csv"  
parse_text_file(input_file, output_csv)