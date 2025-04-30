import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from logger import log_message

def login():
    load_dotenv()
    HUGGING_FACE_HUB_TOKEN=os.environ('HUGGING_FACE_HUB_TOKEN')
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
    if torch.cuda.get_device_capability()[0] < 8:
        log_message(
            {
                "type":"ERROR",
                "text":"Current GPU setup is unavailable for flash attention!",
            }
        )
    

def conversation_format(system_prompt: str, sample):
    return {
        "messages":[
            {"role":"system", "content":system_prompt},
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
        "prompt": [{"role": "user", "content": sample["Question"]}],
        "completion": [{"role": "assistant", "content": sample["Response"]}],
        }