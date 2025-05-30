# **Fine-Tuning Script Guide**  
## **Hydra Configurations Explanation** 

```--model: the name of llm model.```

```--dataset: the link to the dataset. It could be from huggingface, local directory in the format of json or csv.```

```--format: choose the format for the dataset. It could be standard, conversational, preference. More info can be found:https://huggingface.co/docs/trl/en/dataset_formats#formats``` ****

```--trainer: SFT, GRPO, DPO.```****

```--distributed-training: Always default to unsloth.```

```--per-device-train-batch-size: Keep it between 1 or 2 for simplicity. Batch size of one device.```

```--gradient-acc-steps: Gradient accumulation step.```

```--optim: Keep it adamw_torch_fused for simplicity.```

```--learning-rate: Learning rate of the model. Set to 2e-4 for simplicity.```

```--num-train-epochs: The number of times the model get to see every data row.```

```--lr-scheduler-type: Linear schedule type of trainer. Set to cosine for simplicity.```

```--save-steps: Depends on the number of steps to set as checkpoint. If there was an accident, it will be helpful cause the model will start at the checkpoint.```

```--max-completion-length: The maximum of output tokens that the model adapter will generate after finishing training. This depends on you.```

```---max-seq-length 2048: The maximum of input tokens that the model take in. If your single input has more tokens than this, it will reduce.```

```--gradient-checkpointing: Set to True to prevent O-O-M (Out of memory).```

```--report-to: Set to wandb (You should create a Wandb account) to track the model performance!```

## **Distributed Training Methods Option** 
#### To enable FSDP and DDP, you need to set up the accelerate config first, learn more about it at: https://huggingface.co/docs/accelerate/usage_guides/fsdp 




