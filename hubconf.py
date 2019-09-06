from pytorch_transformers import (
    AutoTokenizer, AutoConfig, AutoModel, AutoModelWithLMHead, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
)
from pytorch_transformers.file_utils import add_start_docstrings

dependencies = ['torch', 'tqdm', 'boto3', 'requests', 'regex', 'sentencepiece', 'sacremoses']

@add_start_docstrings(AutoConfig.__doc__)
def config(*args, **kwargs):
    r""" 
                # Using torch.hub !
                import torch

                config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')  # Download configuration from S3 and cache.
                config = torch.hub.load('huggingface/pytorch-transformers', 'config', './test/bert_saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
                config = torch.hub.load('huggingface/pytorch-transformers', 'config', './test/bert_saved_model/my_configuration.json')
                config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased', output_attention=True, foo=False)
                assert config.output_attention == True
                config, unused_kwargs = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased', output_attention=True, foo=False, return_unused_kwargs=True)
                assert config.output_attention == True
                assert unused_kwargs == {'foo': False}

            """

    return AutoConfig.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoTokenizer.__doc__)
def tokenizer(*args, **kwargs):
    r""" 
        # Using torch.hub !
        import torch

        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`

    """

    return AutoTokenizer.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModel.__doc__)
def model(*args, **kwargs):
    r"""
            # Using torch.hub !
            import torch

            model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = torch.hub.load('huggingface/pytorch-transformers', 'model', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = torch.hub.load('huggingface/pytorch-transformers', 'model', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """

    return AutoModel.from_pretrained(*args, **kwargs)

@add_start_docstrings(AutoModelWithLMHead.__doc__)
def modelWithLMHead(*args, **kwargs):
    r"""
        # Using torch.hub !
        import torch

        model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
        model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased', output_attention=True)  # Update configuration during loading
        assert model.config.output_attention == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
        model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    """
    return AutoModelWithLMHead.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForSequenceClassification.__doc__)
def modelForSequenceClassification(*args, **kwargs):
    r"""
            # Using torch.hub !
            import torch

            model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """

    return AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForQuestionAnswering.__doc__)
def modelForQuestionAnswering(*args, **kwargs):
    r"""
        # Using torch.hub !
        import torch

        model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-base-uncased')    # Download model and configuration from S3 and cache.
        model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-base-uncased', output_attention=True)  # Update configuration during loading
        assert model.config.output_attention == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
        model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    """
    return AutoModelForQuestionAnswering.from_pretrained(*args, **kwargs)
