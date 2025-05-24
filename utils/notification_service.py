import os

nvidia_daily_ci_workflow = "huggingface/transformers/.github/workflows/self-scheduled-caller.yml"
amd_daily_ci_workflows = (
    "huggingface/transformers/.github/workflows/self-scheduled-amd-mi210-caller.yml",
    "huggingface/transformers/.github/workflows/self-scheduled-amd-mi250-caller.yml",
)
is_nvidia_daily_ci_workflow = os.environ.get("GITHUB_WORKFLOW_REF").startswith(nvidia_daily_ci_workflow)
is_amd_daily_ci_workflow = os.environ.get("GITHUB_WORKFLOW_REF").startswith(amd_daily_ci_workflows)

print(is_nvidia_daily_ci_workflow)
print(is_amd_daily_ci_workflow)
