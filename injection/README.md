
# Overview

This repository provides a streamlined pipeline for evaluating Large Language Model (LLM) outputs against ground truth answers using both local Hugging Face models and OpenAI's API. The primary workflow includes:

1. Generating model outputs (via various decoding strategies) on a chosen dataset (e.g., GSM8K, MATH-500).
2. Parsing those raw outputs (in `.txt` format) to a structured `.csv`.
3. Leveraging OpenAI GPT to judge correctness (i.e., “correct” vs. “incorrect”) of each response.

This approach allows you to seamlessly test multiple inference methods, obtain raw textual results, convert them to tabular data, and perform automated correctness checks, all within a unified codebase.

---

## Directory Structure

```
my_project/
  ├─ requirements.txt
  ├─ openai_api_key.txt               # Contains your OpenAI API key (plaintext)
  ├─ parser/
  │    └─ parser_code.py
  ├─ evaluator/
  │    └─ correctness_judgement.py
  ├─ model_utils/
  │    ├─ model_config.py
  │    └─ generation_methods.py
  ├─ runner/
  │    └─ experiment_runner.py
  └─ main.py
```

- **requirements.txt**  
  Lists all Python dependencies needed for this project.

- **openai_api_key.txt**  
  A plaintext file containing your OpenAI API key. Make sure not to commit it to any public repository.

- **parser/**  
  - **parser_code.py**  
    Provides functionality to parse raw text output files (with certain formatting conventions) and convert them into CSV format.

- **evaluator/**  
  - **correctness_judgement.py**  
    Uses GPT-based evaluation (OpenAI API) to determine whether the model-generated answer is correct or incorrect, based on the provided question, gold answer, and system prompt.

- **model_utils/**  
  - **model_config.py**  
    Defines a simple configuration loader (`configure_model`) to load a Hugging Face `AutoModelForCausalLM` and corresponding tokenizer.
  - **generation_methods.py**  
    Implements several decoding strategies such as:
    - Greedy Decoding
    - Zero-Shot CoT (Chain-of-Thought)
    - Step Injection
    - Top-K Injection
    - Zero-Shot Next-Step

- **runner/**  
  - **experiment_runner.py**  
    Orchestrates the entire experiment cycle. Provides:
    - Per-method model inference and `.txt` output generation (supports both standard GSM8K-like datasets and MATH-500).
    - Parsing the resulting `.txt` to `.csv`.
    - Integrating OpenAI-based correctness evaluation into a final `.csv` output with an extra column indicating correctness.

- **main.py**  
  - Serves as the CLI entry point. Accepts various command-line arguments to specify dataset, model name, decoding methods, etc., and executes the full pipeline.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/<REPOSITORY_NAME>.git
   cd my_project/
   ```

2. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key:**
   - Insert your secret key into the file `openai_api_key.txt`.  
     For instance:
     ```
     sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
     ```
   - Ensure this file is **excluded** from any public commits (e.g., via `.gitignore`).

---

## Usage

Below is a sample command illustrating how to run the full pipeline:

```bash
python main.py \
    --llm openai \
    --dataset HuggingFaceH4/MATH-500 \
    --model_name meta-llama/Llama-3.1-8B \
    --methods zs_cot step_injection top_k_injection \
    --max_samples 100 \
    --max_length 500 \
    --output_dir ./results
```

### Notable Arguments

- **`--llm`**: Selects the LLM evaluation backend (currently “openai” is supported, leveraging the OpenAI API).
- **`--dataset`**: Specifies which dataset to load via the `datasets` library (e.g., `gsm8k`, `HuggingFaceH4/MATH-500`, or any other compatible dataset identifier).
- **`--model_name`**: Points to the local Hugging Face model (e.g., `meta-llama/Llama-3.1-8B`).
- **`--methods`**: A list of decoding methods to test (`greedy`, `zs_cot`, `step_injection`, `top_k_injection`, `zs_next_step`, etc.).
- **`--max_samples`**: Indicates how many samples from the dataset to process.
- **`--max_length`**: Maximum token length for decoding.
- **`--output_dir`**: Directory where output files (txt/csv) will be saved.

---

## Output Files

After running the pipeline, the specified output directory (e.g., `./results`) will contain files such as:

- **`{MODEL_SHORT_NAME}_{METHOD}.txt`**  
  Raw inference output in text form, structured by blocks.

- **`{MODEL_SHORT_NAME}_{METHOD}.csv`**  
  The CSV version of the `.txt` file, produced by the parser module.

- **`{MODEL_SHORT_NAME}_{METHOD}_evaluated.csv`**  
  The final CSV with an extra `is_correct` column indicating GPT-based correctness.

For instance, if your model name is `meta-llama/Llama-3.1-8B` (shortened to `Llama-3.1-8B`) and the method is `zs_cot`, you might find:
```
results/
  ├─ Llama-3.1-8B_zs_cot.txt
  ├─ Llama-3.1-8B_zs_cot.csv
  └─ Llama-3.1-8B_zs_cot_evaluated.csv
```

---

## Extending & Customizing

- **Custom LLM**: If you wish to evaluate with a different LLM API or a local scoring function, modify `main.py` (the section where `llm` is set) and/or `correctness_judgement.py`.
- **Decoding Parameters**: Each decoding method (`generation_methods.py`) can be tailored with additional hyperparameters (e.g., temperature, top-p, etc.).
- **Dataset Structure**: This pipeline expects a certain schema for standard question-answer problems (e.g., GSM8K) or MATH-500. Adapting other datasets may require minor changes in the `runner/experiment_runner.py` methods.

---

## Notes & Caveats

- **OpenAI API Usage**: This pipeline leverages GPT (via OpenAI API) to classify answers as correct or incorrect. If you do not have an API key, certain functionalities may be limited or require an alternative LLM for evaluation.
- **Resource Constraints**: Larger models (e.g., multi-billion parameter scale) require significant GPU memory. Ensure your environment can handle the model specified in `--model_name`.
- **Data Privacy**: Confirm that sending questions and answers to OpenAI aligns with your privacy or confidentiality requirements.
