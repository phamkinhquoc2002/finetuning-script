import os
from src.configs import TrainingConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import AutoModelForCausalLM
from typing import Union, Optional
from src.logger import log_message
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, SFTConfig, GRPOConfig, DPOTrainer, GRPOTrainer, SFTTrainer
from peft import LoraConfig, get_peft_model

def unpack_training_configuration(config_class:Union[type[DPOConfig], 
                                               type[SFTConfig],
                                               type[GRPOConfig]], 
                                               trainingConfig: TrainingConfig) -> Union[DPOConfig, SFTConfig, GRPOConfig]:
    
    use_reentrant = trainingConfig.distributed_training != "DDP"

    common_params = {
        "output_dir": "./training_checkpoints",
        "per_device_train_batch_size": trainingConfig.per_device_train_batch_size,
        "gradient_accumulation_steps": trainingConfig.gradient_acc_steps,  
        "optim": trainingConfig.optim,
        "save_steps": trainingConfig.save_steps,
        "learning_rate": trainingConfig.learning_rate,
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "logging_steps": 50,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": trainingConfig.lr_scheduler_type,
        "num_train_epochs": trainingConfig.num_train_epochs,
        "gradient_checkpointing": trainingConfig.gradient_checkpointing,
        "report_to": "wandb",
        "gradient_checkpointing_kwargs": {"use_reentrant": use_reentrant}
    }
    if config_class == DPOConfig or config_class == GRPOConfig:
        if hasattr(trainingConfig, "max_completion_length"):
            common_params["max_completion_length"] = trainingConfig.max_completion_length
    elif config_class == SFTConfig and hasattr(trainingConfig, "max_completion_length") and trainingConfig.max_completion_length:
        log_message({
            "type": "INFO",
            "text": "Note: 'max_completion_length' is not used in SFTConfig and will be ignored"
        })
        common_params["max_seq_length"] = trainingConfig.max_sequence_length
    return config_class(**common_params)

    
def model_pack(
        training_config: TrainingConfig):
    
    if training_config.distributed_training == "Unsloth":
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=training_config.model,
        max_seq_length=training_config.max_seq_length,
        max_lora_rank=128,
        load_in_4bit=False,
        gpu_memory_utilization=0.7
    )
        
        model = FastLanguageModel.get_peft_model(
        model=model,
        r=128,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth"
    )
    
    else:
        model, tokenizer = AutoModelForCausalLM.from_pretrained(
            training_config.model,
            torch_dtype="bfloat16",
            device_map=training_config.distributed_training,
            attn_implementation="flash_attention_2",
        )     

        lora_config = LoraConfig(lora_alpha=64,
                                 lora_dropout=0.05,
                                 r=128,
                                 bias="none",
                                 target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"],
                                 task_type="CAUSAL_LM")
        model = get_peft_model(model=model, peft_config=lora_config)


    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

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