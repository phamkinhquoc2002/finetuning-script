import os
import hydra
from pydantic import ValidationError
from src.trainers import unpack_training_configuration, model_pack, last_checkpoint, trainer_setup
from src.reward_functions import citation_reward_function
from src.utils import gpu_compability_check, login
from src.config_models import TrainingConfig, DataConfig
from dataloader.dataloaders import CSVDataLoader, JSONDataLoader, HuggingFaceDataLoader

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    login()
    gpu_compability_check()
    try: 
        training_config = TrainingConfig(
            model = cfg.training_configs.model,
            trainer=cfg.training_configs.trainer,
            distributed_training=cfg.training_configs.distributed_training,
            per_device_train_batch_size=cfg.training_config.per_device_train_batch_size,
            gradient_acc_steps= cfg.training_config.gradient_acc_steps,
            optim= cfg.training_config.optim,
            learning_rate= cfg.training_config.learning_rate,
            num_train_epochs= cfg.training_config.num_train_epochs,
            lr_scheduler_type= cfg.training_config.lr_scheduler_type,
            save_steps= cfg.training_config.save_steps,
            max_completion_length= cfg.training_config.max_completion_length,
            max_seq_length= cfg.training_config.max_seq_length,
            gradient_checkpointing= cfg.training_config.gradient_checkpointing,
            report_to= cfg.training_config.report_to,
            )
        
        data_config = DataConfig(
            data_path=cfg.data.path,
            format=cfg.data.format
        )
    except ValidationError as e:
        raise e
    
    training_args = unpack_training_configuration(
        config_class=training_config.trainer,
        trainingConfig=training_config
    )

    model, tokenizer = model_pack(training_config=training_config)

    if os.path.exists(data_config.path):
        if data_config.path.endswith('.csv'):
            dataset_loader = CSVDataLoader(path=data_config.path,
                                    format=data_config.format)
            dataset = dataset_loader.load()
        elif data_config.dataset.endswith('.json'):
            dataset_loader = JSONDataLoader(path=data_config.path,
                                            format=data_config.format)
            dataset = dataset_loader.load()
    elif not os.path.exists(data_config.dataset):
        dataset_loader = HuggingFaceDataLoader(path=data_config.path, 
                                               format=data_config.format)
        dataset=dataset_loader.load()
    
    if training_config.trainer == 'GRPO':
        funcs = [citation_reward_function]
    else:
        funcs = None

    trainer = trainer_setup(model=model,
                            tokenizer=tokenizer,
                            dataset=dataset,
                            funcs = funcs,
                            config=training_args)
    
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()
    
    checkpoint = last_checkpoint(config=training_args)
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.model.save_pretrained("./adapters")

if __name__ == "__main__":
    main()