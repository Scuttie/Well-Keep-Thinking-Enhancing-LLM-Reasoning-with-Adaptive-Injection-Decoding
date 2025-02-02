import re
import csv

def parse_txt_to_csv(input_path, output_path):
    """
    주어진 txt 파일을 파싱하여 CSV로 저장합니다.
    (CommonsenseQA, GSM8K 등에서 공통 사용 가능하게 수정)

    1) "Test sample #(\d+)" 뿐 아니라 "Sample #(\d+)" 도 잡히도록 정규식 변경
    2) [모범답안] 블록은 없으면 None 처리
    3) CommonsenseQA 방식에서는 "Sample #(\d+)" 아래에
       Question / [정답] / --- [A] final answer --- 식으로만 들어가므로,
       해당 부분이 없어도(예: [모범답안]) 에러 없이 넘어가게끔 처리
    """

    # 1) 텍스트 파일 읽기 + 특정 불필요 문구 제거
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

    # 2) 샘플 블록 정규식:
    #    "Test sample #(\d+)" 또는 "Sample #(\d+)" 로 시작하는 구간을 추출
    pattern_samples = re.compile(
        r'(?s)(?:=){28,}\s*\n(?:Test sample|Sample)\s*#(\d+)(.*?)(?=(?:=){28,}\s*\n(?:Test sample|Sample)\s*#\d+|$)'
    )
    sample_blocks = pattern_samples.findall(text)

    rows = []

    # 3) 각 샘플 블록 파싱
    for sample_num_str, sample_chunk in sample_blocks:
        sample_num = sample_num_str.strip()

        # (A) Question 추출
        #     [모범답안], [정답], --- 등 전/후로 끊기
        q_match = re.search(
            r'(?s)Question:\s*(.*?)(?=\[모범답안\]|\[정답\]|---|$)',
            sample_chunk
        )
        if q_match:
            question_full_text = q_match.group(0)
            question_text = q_match.group(1).strip()
            # 필요시 sample_chunk에서 Question 부분 제거 가능 (원본 코드처럼)
            sample_chunk = sample_chunk.replace(question_full_text, "")
        else:
            question_text = ""

        # (B) [모범답안] (optional)
        gold_match = re.search(
            r'(?s)\[모범답안\]\s*(.*?)(?=\[정답\]|---|$)',
            sample_chunk
        )
        if gold_match:
            gold_answer = gold_match.group(1).strip()
        else:
            gold_answer = ""

        # (C) [정답] 블록에서 "#### ..." 추출
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

        # (D) --- [A/B/C/...] 형태 블록들 파싱
        #     여러 블록이 있을 수도 있으므로 기존 코드대로 정규식 사용
        block_pattern = re.compile(
            r'(?s)---\s*\[([A-Za-z0-9]+)\]\s+(.*?)\s+---\s*(.*?)(?=---\s*\[[A-Za-z0-9]+\]|$)'
        )
        blocks = block_pattern.findall(sample_chunk)

        # 블록이 하나도 안 잡히면(CommonsenseQA처럼 --- [A] final answer ---만 있을 때)
        # 위 정규식이 실패하는 경우가 있으니, 여기서 별도 예외 처리
        # (ex. "final answer"라는 문자열 뒤에 줄바꿈만 있고 추가 "---"가 없으면 패턴이 깨질 수 있음)
        if not blocks:
            # 간단히 '--- [A]' 단일 블록만이라도 잡아보는 정규식
            single_block_match = re.search(
                r'(?s)---\s*\[([A-Za-z0-9]+)\]\s+(.*?)\s+---\s*(.*)',
                sample_chunk
            )
            if single_block_match:
                blocks = [single_block_match.groups()]  # 튜플 하나로
            else:
                # 블록이 전혀 없다면 CSV에 기록할 데이터가 없는 것과 같으므로 넘어감
                # (원하는 로직에 따라 처리)
                pass

        # (E) 블록을 순회하며 CSV 행 생성
        #     CommonsenseQA는 보통 --- [A] final answer --- 하나만 있을 테니
        #     blocks가 1개일 확률이 큼
        if not blocks:
            # 그래도 최소 한 줄을 뽑고 싶다면, 아래처럼 "llm_answer"=None으로 생성 가능
            row = {
                "number": sample_num,
                "type": "",  # 블록 태그 없음
                "question": question_text,
                "gold_answer": gold_answer,
                "answer": answer_text,
                "llm_answer": "",
                "elapsed_time": ""
            }
            rows.append(row)
        else:
            # 여러 블록이 있을 때도 각각 row로 저장
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
                    "elapsed_time": ""
                }
                rows.append(row)

    # 4) CSV 저장
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
