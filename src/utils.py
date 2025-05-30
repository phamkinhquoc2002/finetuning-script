import os
import torch
from dotenv import load_dotenv
# from huggingface_hub import login
from src.logger import log_message

# def hugging_face_login():
#     load_dotenv()
#     HUGGING_FACE_HUB_TOKEN= os.environ['HF_TOKEN']
#     try:
#         login(
#         token=HUGGING_FACE_HUB_TOKEN,
#         add_to_git_credential=True
#     )
#     except Exception as error:
#         log_message(
#             {
#                 "type": "ERROR",
#                 "text": error
#             }
#         )
#         raise

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
    
def conversation_format(sample):
    return {
        "messages":[
            {"role":"system", "content":"You are a personal assistant."},
            {"role":"user", "content":sample["instruction"]},
            {"role":"assistant", "content":sample["output"]}
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

# def flatten_conversation(example):
#     if "messages" in example:
#         messages = example["messages"]
#         convo = ""
#         for m in messages:
#             convo += f"{m['role']}: {m['content']}\n"
#         return {"text": convo.strip()}
#     elif "prompt" in example:
#         convo = f"{example['prompt'][0]['role']}: {example['prompt'][0]['content']}\n{example['completion'][0]['role']}: {example['completion'][0]['content']}"
#         return {"text": convo.strip()}
#     else:
#         return example