import shutil
from pathlib import Path
from huggingface_hub import HfApi

REPO = "dwilson2547/Qwen3.8-27B-IQ3_M"
api = HfApi()

api.create_repo(REPO, repo_type="model", exist_ok=True)
print("repo ready", flush=True)

# card + license first (small, instant feedback)
api.upload_folder(
    repo_id=REPO,
    folder_path="/home/daniel/m0/publish/27b-iq3m-card",
    repo_type="model",
    commit_message="Model card + Apache-2.0 LICENSE (Qwen3.8-27B IQ3_M imatrix quant)",
)
print("card uploaded", flush=True)

# the 12.77 GB quant — resumable, may take a while on the box's uplink
api.upload_file(
    repo_id=REPO,
    path_or_fileobj="/home/daniel/m0/baseline-27b/Qwen3.8-27B-IQ3_M.gguf",
    path_in_repo="Qwen3.8-27B-IQ3_M.gguf",
    repo_type="model",
    commit_message="IQ3_M imatrix quant (11.9 GiB)",
)
print("PUBLISH_27B_DONE", flush=True)
