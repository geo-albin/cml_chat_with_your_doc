
from huggingface_hub import snapshot_download

def download_snapshot(repo_id):
    res = snapshot_download(repo_id=repo_id,local_dir="huggingface_mirror")
    print(f"Albin snapshot download - {res}")

def download_embed(repo_id):
    res = snapshot_download(repo_id=repo_id,local_dir="huggingface_mirror")
    print(f"Albin embed download - {res}")

if __name__ == "__main__":
    download_snapshot("mistralai/Mistral-7B-Instruct-v0.2")
    download_embed("mixedbread-ai/mxbai-embed-large-v1")