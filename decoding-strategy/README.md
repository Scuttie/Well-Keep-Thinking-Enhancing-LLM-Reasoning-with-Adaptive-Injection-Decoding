# Decoding Strategies

| **File**                | **Description**                                                                 | 
|-------------------------|---------------------------------------------------------------------------------|
| `cot-decoding.py`       | 'Chain-of-Thought Reasoning Without Prompting' based decoding|
| `eos-decoding.py`       | Continues generating text even after encountering the `<eos>` token, up to `max_length`. |
| `first-sentence.py`     | Generates the first sentences in cot w/o prompting manner, then expands to full responses using the input and the initial sentence.| 
