# **Fine-Tuning Script Guide**  

## **Running Command Example**  

To fine-tune a model, run the following command:  

```
python train.py --model "meta-llama/Llama-3.2-1B" --dataset "FreedomIntelligence/medical-o1-reasoning-SFT" --format "conversational" --trainer "SFT" --distributed-training "Unsloth" --per-device-train-batch-size 1 --gradient-acc-steps 4 --optim "adamw_torch_fused" --learning-rate 2e-4 --num-train-epochs 1 --lr-scheduler-type "cosine" --save-steps 50 --max-completion-length 1024 --max-seq-length 2048 --gradient-checkpointing True --report-to "wandb"
 ```

## **Arguments Explanation** 

```--model: the name of llm model.```

```--dataset: the link to the dataset. It could be from huggingface, local directory in the format of json or csv.```

```--format: choose the format for the dataset. It could be standard, conversational, preference. More info can be found:https://huggingface.co/docs/trl/en/dataset_formats#formats``` ****

```--trainer: SFT, GRPO, DPO.```****

```--distributed-training: Always default to unsloth since it is the fastest.```

```--per-device-train-batch-size: Keep it between 1 or 2 for simplicity. Batch size of one device.```

```--gradient-acc-steps: Gradient accumulation step.```

```--optim: Keep it adamw_torch_fused for.```

```--learning-rate: Learning rate of the model. Set to 2e-4 for simplicity.```

```--num-train-epochs: The number of times the model get to see every data row.```

```--lr-scheduler-type: Linear schedule type of trainer. Set to cosine for simplicity.```

```--save-steps: Depends on the number of steps to set as checkpoint. If there was an accident, it will be helpful cause the model will start at the checkpoint.```

```--max-completion-length: The maximum of output tokens that the model adapter will generate after finishing training. This depends on you.```

```---max-seq-length 2048: The maximum of input tokens that the model take in. If your single input has more tokens than this, it will reduce.```

```--gradient-checkpointing: Set to True to prevent O-O-M (Out of memory).```

```--report-to: Set to wandb (You should create a Wandb account) to track the model performance!```

## **Training Methods Option** 
### SFT
- Can be used to train every types of data, except for preference. It will help the model learn better but you need high-quality data.
### DPO
- Best to use when you want to train tasks that need human alignment like summarization or conversational chatbot.
- The hardest to setup cause you need both chosen and rejected responses (good and bad responses). Because of that, it is the most costly method. You can only use the preference data format to train the model.
### GRPO
- Often used if you want a structured output like <citation></citation> or <think></think>. To train this, you just need a set of prompts and a reward function that defines how the model is gonna learn.




