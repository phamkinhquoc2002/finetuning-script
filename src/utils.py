import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from src.logger import log_message

def hugging_face_login():
    load_dotenv()
    HUGGING_FACE_HUB_TOKEN= os.environ['HF_TOKEN']
    try:
        login(
        token=HUGGING_FACE_HUB_TOKEN,
        add_to_git_credential=True
    )
    except Exception as error:
        log_message(
            {
                "type": "ERROR",
                "text": error
            }
        )
        raise

def gpu_compability_check():
    if torch.cuda.is_available():
        log_message(
            {
                "type":"INFO",
                "text": "CUDA AVAILABLE"
            }
        )
    else:
        log_message(
            {
                "type":"ERROR",
                "text":"No GPU available!",
            }
        )
        raise SyntaxError("Environment failed to recognize CUDA!")
    
def conversation_format(sample):
    return {
        "messages":[
            {"role":"system", "content":"You are a personal assistant."},
            {"role":"user", "content":sample["Question"]},
            {"role":"assistant", "content":sample["Response"]}
        ]
    }

def preference_format(sample):
    return {
        "prompt": [{"role": "user", "content": sample["Question"]}],
        "chosen": [{"role": "assistant", "content": sample["Chosen"]}],
        "rejected": [{"role": "assistant", "content": sample["Rejected"]}]
        }

def standard_format(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["Response"]},
            {"role": "assistant", "content": sample["Question"]},
        ]
    }

def is_valid_sample(example):
    return example.get("Response") is not None