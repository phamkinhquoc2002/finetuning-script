from pydantic import BaseModel, Field
from typing import Literal

class TrainingConfig(BaseModel):
    """

    """
    model: str = Field(default="./my_saved_model")
    trainer: Literal["SFT", "DPO", "GRPO"]
    distributed_training: Literal["DDP", "FSDP", "Unsloth"]
    per_device_train_batch_size: int = Field(default=2)
    gradient_acc_steps: int = Field(default=4)
    optim: str = Field(default="adamw_torch_fused")
    learning_rate: float = Field(default=5e-5)
    num_train_epochs: int = Field(default=1)
    lr_scheduler_type: str = Field(default="cosine")
    save_steps: int = Field(default=100)
    max_completion_length: int = Field(default=2048)
    max_seq_length: int = Field(defaul=4096)
    gradient_checkpointing: bool = Field(default=True)
    report_to: str = Field(default="tensorboard")

class DataConfig():
    """
    """
    data_path: str = Field(default="./data")
    format: Literal["no", "conversational", "preference", "standard"]
    