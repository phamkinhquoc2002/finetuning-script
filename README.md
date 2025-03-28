# FINETUNING SCRIPT GUIDELINE

### Running Command Example:

```python train.py --model "meta-llama/Llama-3.2-1B" --dataset "FreedomIntelligence/medical-o1-reasoning-SFT" --format "conversational" --trainer "SFT" --distributed-training "Unsloth" --per-device-train-batch-size 1 --gradient-acc-steps 4 --optim "adamw_torch_fused" --learning-rate 2e-4 --num-train-epochs 1 --lr-scheduler-type "cosine" --save-steps 50 --max-completion-length 1024 --max-seq-length 2048 --gradient-checkpointing True --report-to "wandb"```


ERRORS

1. If it returned the error



* You might not have the updated cddn version on your VM/GPU setup. Please download it this way: 

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz

* And then extract it by:

tar -xvf cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz
