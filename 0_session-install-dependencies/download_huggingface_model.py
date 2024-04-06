from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

MODELS_PATH = "./models"
EMBEDS_PATH = "./embed_models"

model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    resume_download=True,
    cache_dir=MODELS_PATH,
    local_files_only=False,
)

embed_model = "thenlper/gte-large"

model_path = snapshot_download(
    repo_id=embed_model,
    resume_download=True,
    cache_dir=EMBEDS_PATH,
    local_files_only=False,
)
