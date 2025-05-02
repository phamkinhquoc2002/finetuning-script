from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="Qwen/Qwen3-14B",  
    local_dir="./my_saved_model",             
    local_dir_use_symlinks=False
)
                   
print("Model downloaded to:", local_dir)