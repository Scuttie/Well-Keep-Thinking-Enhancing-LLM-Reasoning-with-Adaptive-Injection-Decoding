import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def configure_model(model_name: str):
    """
    주어진 model_name에 따라 모델과 토크나이저를 로드하고 반환합니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
