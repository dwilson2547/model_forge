import shutil
from huggingface_hub import HfApi

api = HfApi()
MAIN = "dwilson2547/Qwen3.5-4B-drop2-mixed-repair"
GGUF = "dwilson2547/Qwen3.5-4B-drop2-mixed-repair-GGUF"

api.create_repo(MAIN, repo_type="model", exist_ok=True)
api.create_repo(GGUF, repo_type="model", exist_ok=True)
print("repos created", flush=True)

api.upload_folder(repo_id=MAIN, folder_path="/home/daniel/m0/publish/mixfull",
                  repo_type="model", commit_message="Model: drop2 slice + mixed-diet QLoRA repair (model-forge)")
print("main repo uploaded", flush=True)

api.upload_file(
    repo_id=GGUF,
    path_or_fileobj="/home/daniel/m0/lmstudio-out/mixfull/mixfull-Q4_K_M-mtp.gguf",
    path_in_repo="Qwen3.5-4B-drop2-mixed-repair-Q4_K_M.gguf",
    repo_type="model",
    commit_message="Q4_K_M with MTP block (draft tensors pre-slice)",
)
print("gguf repo uploaded", flush=True)
print("PUBLISH_DONE", flush=True)
