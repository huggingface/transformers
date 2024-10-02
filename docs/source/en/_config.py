# docstyle-ignore
INSTALL_CONTENT = """
# Transformers installation
! pip install transformers datasets evaluate accelerate
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
black_avoid_patterns = {
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",
}

extensions = [
    'sphinxcontrib.redirects'
]

redirects = {
    "components/training/pytorch": "https://www.kubeflow.org/docs/components/training/user-guides/pytorch/",
    "components/starter/install": "https://www.kubeflow.org/docs/started/installing-kubeflow/",
    "components/starter/kubectl": "https://kubernetes.io/docs/tasks/tools/",
    "components/starter/volumeclaim": "https://kubernetes.io/docs/concepts/storage/persistent-volumes/",
    "components/starter/storage": "https://kubernetes.io/docs/concepts/storage/storage-classes/",
    "components/cpuunits": "https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu",
    "components/guaranteed": "https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/#guaranteed",
    "components/qualityofservice": "https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod/"
}