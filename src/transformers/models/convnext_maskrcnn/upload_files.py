from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi


api = HfApi()
operations = [
    CommitOperationAdd(path_in_repo="neg_bboxes_roi.pt", path_or_fileobj="./neg_bboxes_roi.pt"),
]
api.create_commit(repo_id="nielsr/init-files", operations=operations, commit_message="Upload neg_bboxes_roi.pt")
