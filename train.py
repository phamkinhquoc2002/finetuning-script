import argparse
import os
from utils.trainers import gpu_compability_check, login, unpack_training_configuration, model_pack, last_checkpoint, trainer_setup
from utils.reward_functions import citation_reward_function
from dataloader.dataloaders import CSVDataLoader, JSONDataLoader, HuggingFaceDataLoader

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the Large Language Model training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Large language Model Training Script"
        )
    parser.add_argument("--model", 
                        type=str, 
                        help="Model Name")
    parser.add_argument("--dataset", 
                        type=str, 
                        help="The path link to where the dataset locates")
    parser.add_argument("--format", 
                        type=str, 
                        choices=["conversational", "standard", "preference", "no"],
                        default="no",
                        help="Dataset Format Function if needed")
    parser.add_argument("--trainer", 
                        type=str, 
                        choices=["SFT", "GRPO", "DPO"],
                        default="SFT",
                        help="Type of Trainers: GRPO, SFT, DPO")
    parser.add_argument("--distributed-training", 
                        type=str, 
                        choices=["DDP", "FSDP", "Unsloth"],
                        default="DDP",
                        help="Distributed Training Methods: DDP, FSDP")
    parser.add_argument("--per-device-train-batch-size",
                        type=int,
                        default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient-acc-steps",
                        type=int,
                        default=4,
                        help="Gradient accumulation steps!")
    parser.add_argument("--optim",
                        type=str, 
                        default="adamw_torch_fused",
                        help="Optimizer Methods!")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=5e-5,
                        help="The training learning rate")
    parser.add_argument("--num-train-epochs",
                        type=int,
                        default=2,
                        help="The total number of epochs")
    parser.add_argument("--lr-scheduler-type",
                        type=str,
                        default="cosine",
                        help="lr schedule function")  
    parser.add_argument("--save-steps",
                        type=int,
                        default=100,
                        help="The number of steps to save a checkpoint")
    parser.add_argument("--max-completion-length",
                        type=int,
                        default=2048,
                        help="Total number of completion tokens")
    parser.add_argument("--max-seq-length",
                        type=int,
                        default=2048,
                        help="Total number of sequence tokens")
    parser.add_argument("--gradient-checkpointing",
                        type=bool,
                        default=True,
                        help="Gradient Checkpointing")
    parser.add_argument("--report-to",
                        type=str,
                        default="wandb",
                        help="Visualize training step")

    return parser.parse_args()

if __name__ == '__main__':
    login()
    gpu_compability_check()
    args = parse_arguments()

    training_args = unpack_training_configuration(
        config_class=args.trainer,
        args=args
    )

    model, tokenizer = model_pack(args=args)

    if os.path.exists(args.dataset):
        if args.dataset.endswith('.csv'):
            dataset_loader = CSVDataLoader(path=args.dataset,
                                    format=args.format)
            dataset = dataset_loader.load()
        elif args.dataset.endswith('.json'):
            dataset_loader = JSONDataLoader(path=args.dataset,
                                            format=args.format)
            dataset = dataset_loader.load()
    elif not os.path.exists(args.dataset):
        dataset_loader = HuggingFaceDataLoader(path=args.dataset, 
                                               format=args.format)
        dataset=dataset_loader.load()
    
    if args.trainer == 'GRPO':
        funcs = [citation_reward_function]
    else:
        funcs = None
    trainer = trainer_setup(model=model,
                            tokenizer=tokenizer,
                            dataset=dataset,
                            funcs = None,
                            config=training_args)
    
    checkpoint = last_checkpoint(config=training_args)
    trainer.train(resume_from_checkpoint=last_checkpoint)