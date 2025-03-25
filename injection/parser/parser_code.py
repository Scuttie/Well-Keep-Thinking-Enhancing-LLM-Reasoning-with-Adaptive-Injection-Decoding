import re
import csv
import pandas as pd

def parse_txt_to_csv(input_path, output_path):
    """
    주어진 txt 파일을 파싱하여 CSV로 저장하고,
    추가로 XLSX 파일도 생성한다.
    """

    skip_keywords = [
        "You are a helpful assistant",
        "If you have finished your reasoning",
        "Reason carefully then answer"
    ]

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        if any(k in line for k in skip_keywords):
            continue
        cleaned_lines.append(line)

    text = "".join(cleaned_lines)

    pattern_samples = re.compile(
        r'(?s)(?:=){28,}\s*\n(?:Test sample|Sample)\s*#(\d+)(.*?)(?=(?:=){28,}\s*\n(?:Test sample|Sample)\s*#\d+|$)'
    )
    sample_blocks = pattern_samples.findall(text)

    rows = []

    for sample_num_str, sample_chunk in sample_blocks:
        sample_num = sample_num_str.strip()

        q_match = re.search(
            r'(?s)Question:\s*(.*?)(?=\[모범답안\]|\[정답\]|---|Time taken:|$)',
            sample_chunk
        )
        if q_match:
            question_full_text = q_match.group(0)
            question_text = q_match.group(1).strip()
            sample_chunk = sample_chunk.replace(question_full_text, "")
        else:
            question_text = ""

        gold_match = re.search(
            r'(?s)\[모범답안\]\s*(.*?)(?=\[정답\]|---|Time taken:|$)',
            sample_chunk
        )
        if gold_match:
            gold_answer_full = gold_match.group(0)
            gold_answer = gold_match.group(1).strip()
            sample_chunk = sample_chunk.replace(gold_answer_full, "")
        else:
            gold_answer = ""

        answer_text = ""
        answer_block_match = re.search(
            r'(?s)\[정답\]\s*(.*?)(?=---|Time taken:|$)',
            sample_chunk
        )
        if answer_block_match:
            block_str = answer_block_match.group(1)
            ans_match = re.search(r'####\s*(\S+)', block_str)
            if ans_match:
                answer_text = ans_match.group(1).strip()
            sample_chunk = sample_chunk.replace(answer_block_match.group(0), "")

        time_match = re.search(r'Time taken:\s*([\d\.]+)', sample_chunk)
        if time_match:
            elapsed_time_str = time_match.group(1).strip()
            sample_chunk = sample_chunk.replace(time_match.group(0), "")
        else:
            elapsed_time_str = ""

        block_pattern = re.compile(
            r'(?s)---\s*\[([A-Za-z0-9]+)\]\s+(.*?)\s+---\s*(.*?)(?=---\s*\[[A-Za-z0-9]+\]|Time taken:|$)'
        )
        blocks = block_pattern.findall(sample_chunk)

        if not blocks:
            single_block_match = re.search(
                r'(?s)---\s*\[([A-Za-z0-9]+)\]\s+(.*?)\s+---\s*(.*)',
                sample_chunk
            )
            if single_block_match:
                blocks = [single_block_match.groups()]

        if not blocks:
            row = {
                "number": sample_num,
                "type": "",
                "question": question_text,
                "gold_answer": gold_answer,
                "answer": answer_text,
                "llm_answer": "",
                "elapsed_time": elapsed_time_str
            }
            rows.append(row)
        else:
            for (block_tag, block_label, block_content) in blocks:
                block_type = f"[{block_tag}] {block_label}".strip()
                llm_answer = block_content.strip()

                row = {
                    "number": sample_num,
                    "type": block_type,
                    "question": question_text,
                    "gold_answer": gold_answer,
                    "answer": answer_text,
                    "llm_answer": llm_answer,
                    "elapsed_time": elapsed_time_str
                }
                rows.append(row)

    # CSV 저장
    fieldnames = [
        "number",
        "type",
        "question",
        "gold_answer",
        "answer",
        "llm_answer",
        "elapsed_time"
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Parsed results are saved in {output_path}")

    df = pd.read_csv(output_path)
    excel_path = output_path.replace(".csv", ".xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Excel file saved -> {excel_path}")

