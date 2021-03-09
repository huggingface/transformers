import os
import pytest

# TODO add in setup.py there is a release 

# | bert-finetuning-pytorch       | testbertfinetuningusingBERTfromtransformerlib+PT                  | SageMaker createTrainingJob | 1        | Accuracy, time to train |
# |-------------------------------|-------------------------------------------------------------------|-----------------------------|----------|-------------------------|
# | bert-finetuning-pytorch-ddp   | test bert finetuning using BERT from transformer lib+ PT DPP      | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |
# | bert-finetuning-pytorch-smddp | test bert finetuning using BERT from transformer lib+ PT SM DDP   | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |
# | deberta-finetuning-smmp       | test deberta finetuning using BERT from transformer lib+ PT SM MP | SageMaker createTrainingJob | 2/4/8/16 | Accuracy, time to train |
@pytest.mark.skipif(os.environ["TEST_SAGEMAKER"] != True, reason="Skipping test because should only be run when releasing minor transformers version")
@pytest.mark.parametrize('model_name_or_path', ["distilbert-base-uncased"])
@pytest.mark.parametrize('instance_types', ["ml.p3dn.24xlarge"])
@pytest.mark.parametrize('instance_count', [2,4,8,16])
def test_single_node_fine_tuning(instance_types, instance_count, model_name_or_path):
  print(instance_types)
  print(instance_count)
  model_name_or_path



@pytest.mark.skipif(os.environ["TEST_SAGEMAKER"] != True, reason="Skipping test because should only be run when releasing minor transformers version")
@pytest.mark.parametrize('model_name_or_path', ["distilbert-base-uncased"])
@pytest.mark.parametrize('instance_types', ["ml.p3dn.24xlarge"])
@pytest.mark.parametrize('instance_count', [2,4,8,16])
def test_multi_node_fine_tuning(instance_types, instance_count, model_name_or_path):
    """
    Tests smddprun command via Estimator API distribution parameter
    """
    pass
    # distribution = {"smdistributed":{"dataparallel":{"enabled":True}}}
    # estimator = PyTorch(entry_point='smdataparallel_mnist.py',
    #                     role='SageMakerRole',
    #                     image_uri=ecr_image,
    #                     source_dir=mnist_path,
    #                     instance_count=2,
    #                     instance_type=instance_types,
    #                     sagemaker_session=sagemaker_session,
    #                     debugger_hook_config=False,
    #                     distribution=distribution)

    # estimator.fit()