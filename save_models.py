import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_file(
    path_or_fileobj="models/query_tower.pt", 
    path_in_repo="query_tower.pt",   
    repo_id="amyf/ms-marco",
    repo_type="model",
)
api.upload_file(
    path_or_fileobj="models/doc_tower.pt", 
    path_in_repo="doc_tower.pt",   
    repo_id="amyf/ms-marco",
    repo_type="model",
)