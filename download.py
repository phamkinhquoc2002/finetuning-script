from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="llm-jp/llm-jp-3-13b-instruct3",  
    local_dir="./my_saved_model/llm-jp-3-13b-instruct3",             
    local_dir_use_symlinks=False
)
                   
print("Model downloaded to:", local_dir)