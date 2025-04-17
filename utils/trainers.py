import os
import torch
import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from dotenv import load_dotenv
from huggingface_hub import login
from typing import Union, Optional
from utils.logger import log_message
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, SFTConfig, GRPOConfig, DPOTrainer, GRPOTrainer, SFTTrainer

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

def unpack_training_configuration(config_class:Union[type[DPOConfig], 
                                               type[SFTConfig],
                                               type[GRPOConfig]], 
                                               args: argparse.Namespace) -> Union[DPOConfig, SFTConfig, GRPOConfig]:
    
    use_reentrant = args.distributed_training != "DDP"

    common_params = {
        "output_dir": "./training_checkpoints",
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_acc_steps,  
        "optim": args.optim,
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "logging_steps": 50,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_train_epochs": args.num_train_epochs,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "wandb",
        "gradient_checkpointing_kwargs": {"use_reentrant": use_reentrant}
    }
    if config_class == DPOConfig or config_class == GRPOConfig:
        if hasattr(args, "max_completion_length"):
            common_params["max_completion_length"] = args.max_completion_length
    elif config_class == SFTConfig and hasattr(args, "max_completion_length") and args.max_completion_length:
        log_message({
            "type": "INFO",
            "text": "Note: 'max_completion_length' is not used in SFTConfig and will be ignored"
        })
        common_params["max_seq_length"] = args.max_sequence_length
    return config_class(**common_params)

    
def model_pack(
        args: argparse.Namespace):
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        max_lora_rank=128,
        load_in_4bit=False,
        gpu_memory_utilization=0.7
    )

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=128,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth"
    )

    return model, tokenizer

def last_checkpoint(config: Union[DPOConfig, 
                                  SFTConfig, 
                                  GRPOConfig]):
    last_checkpoint = None
    if os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)
    log_message(
        {
            "type":"INFO",
            "text":"Checkpoint detected {last_checkpoint}. We will continue at this checkpoint!"
        }
    )
    return last_checkpoint

def trainer_setup(model, 
          tokenizer,
          dataset,
          funcs: Optional[list],
          config: Union[GRPOConfig, 
                        DPOConfig, 
                        SFTConfig]) -> Union[GRPOTrainer,
                                             DPOTrainer,
                                             SFTTrainer]:
    if isinstance(config, GRPOConfig):
        if funcs is None:
            log_message(
                {
                    "type":"ERROR",
                    "text": "You need to provide at least one reward function!"
                }
            )
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=funcs,
                args=config,
                train_dataset=dataset
                )
    elif isinstance(config, SFTConfig):
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            dataset=dataset,
            args=config
        )
    elif isinstance(config, DPOConfig):
        trainer = DPOTrainer(
            model=model,
            processing_class=tokenizer,
            dataset=dataset,
            args=config
        )

    log_message(
        {
            "type":"INFO",
            "text":"Trainer has been on set!"
        }
    )
    return trainer