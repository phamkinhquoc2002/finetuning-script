import hydra
from huggingface_hub import snapshot_download

@hydra.main(version_base="1.3", config_path="./configs", config_name="config")
def download(cfg):
    local_dir = snapshot_download(
        repo_id=cfg.evaluation.repo_id,  
        local_dir=cfg.evaluation.local_dir,             
        local_dir_use_symlinks=False
        )                 
    print("Model downloaded to:", cfg.evaluation.local_dir)

if __name__=="__main__":
    download()