import re
import csv

def parse_txt_to_csv(input_path, output_path):
    """
    주어진 txt 파일을 파싱하여 CSV로 저장합니다.

    * 수정 사항:
      - Question 부분을 추출 후 sample_chunk에서 제거하여,
        이후 llm_answer 파트에서 Question이 노출되지 않도록 함.
    """

    # 1) 텍스트 파일 전체를 읽어서, 특정 불필요 문구가 들어있는 줄 제거
    skip_keywords = [
        "You are a helpful assistant",
        "If you have finished your reasoning",
        "Let's think step by step",
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

    # 2) 28개 이상의 '='로 구분되는 샘플 블록을 찾는다.
    pattern_samples = re.compile(
        r'(?s)(?:=){28,}\s*\nTest sample #(\d+)(.*?)(?=(?:=){28,}\s*\nTest sample #\d+|$)'
    )
    test_samples = pattern_samples.findall(text)

    rows = []

    # 각 샘플 블록 파싱
    for sample_num_str, sample_chunk in test_samples:
        sample_num = sample_num_str.strip()

        # (1) Question 추출
        q_match = re.search(
            r'(?s)Question:\s*(.*?)(?=\[모범답안\]|\[정답\]|---|$)',
            sample_chunk
        )

        if q_match:
            question_full_text = q_match.group(0)
            question_text = q_match.group(1).strip()

            # Question 전체를 sample_chunk에서 제거 -> 이후 블록(llm_answer) 파싱 시 Question 제외
            sample_chunk = sample_chunk.replace(question_full_text, "")
        else:
            question_text = ""

        # (2) [모범답안] 블록 추출
        gold_match = re.search(
            r'(?s)\[모범답안\]\s*(.*?)(?=\[정답\]|---|$)',
            sample_chunk
        )
        gold_answer = gold_match.group(1).strip() if gold_match else ""

        # (3) [정답] 블록에서 "#### 숫자" 추출
        answer_text = ""
        answer_block_match = re.search(
            r'(?s)\[정답\]\s*(.*?)(?=---|$)',
            sample_chunk
        )
        if answer_block_match:
            block_str = answer_block_match.group(1)
            ans_match = re.search(r'####\s*(\S+)', block_str)
            if ans_match:
                answer_text = ans_match.group(1).strip()

        # (4) --- [A/B/C/...] 형태 블록들 파싱
        block_pattern = re.compile(
            r'(?s)---\s*\[([A-Za-z0-9]+)\]\s+(.*?)\s+---\s*(.*?)(?=---\s*\[[A-Za-z0-9]+\]|$)'
        )
        blocks = block_pattern.findall(sample_chunk)

        # (5) elapsed_time (필요 시 추가 가능)
        elapsed_time = ""

        if not blocks:
            # 블록이 없는 경우는 무시(혹은 다른 방식 처리 가능)
            continue

        # (6) 블록들을 순회하며 CSV row 생성
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
                "elapsed_time": elapsed_time
            }
            rows.append(row)

    # 3) CSV 저장
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
