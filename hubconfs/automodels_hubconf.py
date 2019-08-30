from pytorch_transformers import (
    AutoTokenizer, AutoConfig, AutoModel, AutoModelWithLMHead, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
)


def autoConfig(*args, **kwargs):
    r""" Instantiates one of the configuration classes of the library
            from a pre-trained model configuration.

            The configuration class to instantiate is selected as the first pattern matching
            in the `pretrained_model_name_or_path` string (in the following order):
                - contains `bert`: BertConfig (Bert model)
                - contains `openai-gpt`: OpenAIGPTConfig (OpenAI GPT model)
                - contains `gpt2`: GPT2Config (OpenAI GPT-2 model)
                - contains `transfo-xl`: TransfoXLConfig (Transformer-XL model)
                - contains `xlnet`: XLNetConfig (XLNet model)
                - contains `xlm`: XLMConfig (XLM model)
                - contains `roberta`: RobertaConfig (RoBERTa model)

            Params:
                pretrained_model_name_or_path: either:

                    - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g.: ``bert-base-uncased``.
                    - a path to a `directory` containing a configuration file saved using the :func:`~pytorch_transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                    - a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.

                cache_dir: (`optional`) string:
                    Path to a directory in which a downloaded pre-trained model
                    configuration should be cached if the standard cache should not be used.

                kwargs: (`optional`) dict: key/value pairs with which to update the configuration object after loading.

                    - The values in kwargs of any keys which are configuration attributes will be used to override the loaded values.
                    - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.

                force_download: (`optional`) boolean, default False:
                    Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

                proxies: (`optional`) dict, default None:
                    A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                    The proxies are used on each request.

                return_unused_kwargs: (`optional`) bool:

                    - If False, then this function returns just the final configuration object.
                    - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part of kwargs which has not been used to update `config` and is otherwise ignored.

            Examples::

                config = AutoConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
                config = AutoConfig.from_pretrained('./test/bert_saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
                config = AutoConfig.from_pretrained('./test/bert_saved_model/my_configuration.json')
                config = AutoConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
                assert config.output_attention == True
                config, unused_kwargs = AutoConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                                   foo=False, return_unused_kwargs=True)
                assert config.output_attention == True
                assert unused_kwargs == {'foo': False}

            """

    return AutoConfig.from_pretrained(*args, **kwargs)


def autoTokenizer(*args, **kwargs):
    r""" Instantiates one of the tokenizer classes of the library
    from a pre-trained model vocabulary.

    The tokenizer class to instantiate is selected as the first pattern matching
    in the `pretrained_model_name_or_path` string (in the following order):
        - contains `bert`: BertTokenizer (Bert model)
        - contains `openai-gpt`: OpenAIGPTTokenizer (OpenAI GPT model)
        - contains `gpt2`: GPT2Tokenizer (OpenAI GPT-2 model)
        - contains `transfo-xl`: TransfoXLTokenizer (Transformer-XL model)
        - contains `xlnet`: XLNetTokenizer (XLNet model)
        - contains `xlm`: XLMTokenizer (XLM model)
        - contains `roberta`: RobertaTokenizer (XLM model)

    Params:
        pretrained_model_name_or_path: either:

            - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.: ``bert-base-uncased``.
            - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~pytorch_transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
            - (not applicable to all derived classes) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.

        cache_dir: (`optional`) string:
            Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

        force_download: (`optional`) boolean, default False:
            Force to (re-)download the vocabulary files and override the cached versions if they exists.

        proxies: (`optional`) dict, default None:
            A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
            The proxies are used on each request.

        inputs: (`optional`) positional arguments: will be passed to the Tokenizer ``__init__`` method.

        kwargs: (`optional`) keyword arguments: will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``. See parameters in the doc string of :class:`~pytorch_transformers.PreTrainedTokenizer` for details.

    Examples::

        config = AutoTokenizer.from_pretrained('bert-base-uncased')    # Download vocabulary from S3 and cache.
        config = AutoTokenizer.from_pretrained('./test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`

    """

    return AutoTokenizer.from_pretrained(*args, **kwargs)


def autoModel(*args, **kwargs):
    r""" Instantiates one of the base model classes of the library
        from a pre-trained model configuration.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `roberta`: RobertaModel (RoBERTa model)
            - contains `bert`: BertModel (Bert model)
            - contains `openai-gpt`: OpenAIGPTModel (OpenAI GPT model)
            - contains `gpt2`: GPT2Model (OpenAI GPT-2 model)
            - contains `transfo-xl`: TransfoXLModel (Transformer-XL model)
            - contains `xlnet`: XLNetModel (XLNet model)
            - contains `xlm`: XLMModel (XLM model)

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
            To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~pytorch_transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and :func:`~pytorch_transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~pytorch_transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModel.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """

    return AutoModel.from_pretrained(*args, **kwargs)


def autoModelWithLMHead(*args, **kwargs):
    r""" Instantiates one of the language modeling model classes of the library
    from a pre-trained model configuration.

    The `from_pretrained()` method takes care of returning the correct model class instance
    using pattern matching on the `pretrained_model_name_or_path` string.

    The model class to instantiate is selected as the first pattern matching
    in the `pretrained_model_name_or_path` string (in the following order):
        - contains `roberta`: RobertaForMaskedLM (RoBERTa model)
        - contains `bert`: BertForMaskedLM (Bert model)
        - contains `openai-gpt`: OpenAIGPTLMHeadModel (OpenAI GPT model)
        - contains `gpt2`: GPT2LMHeadModel (OpenAI GPT-2 model)
        - contains `transfo-xl`: TransfoXLLMHeadModel (Transformer-XL model)
        - contains `xlnet`: XLNetLMHeadModel (XLNet model)
        - contains `xlm`: XLMWithLMHeadModel (XLM model)

    The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
    To train the model, you should first set it back in training mode with `model.train()`

    Params:
        pretrained_model_name_or_path: either:

            - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
            - a path to a `directory` containing model weights saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
            - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

        model_args: (`optional`) Sequence of positional arguments:
            All remaning positional arguments will be passed to the underlying model's ``__init__`` method

        config: (`optional`) instance of a class derived from :class:`~pytorch_transformers.PretrainedConfig`:
            Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

            - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
            - the model was saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
            - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

        state_dict: (`optional`) dict:
            an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
            This option can be used if you want to create a model from a pretrained configuration but load your own weights.
            In this case though, you should check if using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and :func:`~pytorch_transformers.PreTrainedModel.from_pretrained` is not a simpler option.

        cache_dir: (`optional`) string:
            Path to a directory in which a downloaded pre-trained model
            configuration should be cached if the standard cache should not be used.

        force_download: (`optional`) boolean, default False:
            Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

        proxies: (`optional`) dict, default None:
            A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
            The proxies are used on each request.

        output_loading_info: (`optional`) boolean:
            Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

        kwargs: (`optional`) Remaining dictionary of keyword arguments:
            Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

            - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
            - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~pytorch_transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

    Examples::

        model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
        model = AutoModelWithLMHead.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = AutoModelWithLMHead.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
        assert model.config.output_attention == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
        model = AutoModelWithLMHead.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    """
    return AutoModelWithLMHead.from_pretrained(*args, **kwargs)


def autoModelForSequenceClassification(*args, **kwargs):
    r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `roberta`: RobertaForSequenceClassification (RoBERTa model)
            - contains `bert`: BertForSequenceClassification (Bert model)
            - contains `xlnet`: XLNetForSequenceClassification (XLNet model)
            - contains `xlm`: XLMForSequenceClassification (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~pytorch_transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and :func:`~pytorch_transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~pytorch_transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """

    return AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)


def autoModelForQuestionAnswering(*args, **kwargs):
    r""" Instantiates one of the question answering model classes of the library
    from a pre-trained model configuration.

    The `from_pretrained()` method takes care of returning the correct model class instance
    using pattern matching on the `pretrained_model_name_or_path` string.

    The model class to instantiate is selected as the first pattern matching
    in the `pretrained_model_name_or_path` string (in the following order):
        - contains `bert`: BertForQuestionAnswering (Bert model)
        - contains `xlnet`: XLNetForQuestionAnswering (XLNet model)
        - contains `xlm`: XLMForQuestionAnswering (XLM model)

    The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
    To train the model, you should first set it back in training mode with `model.train()`

    Params:
        pretrained_model_name_or_path: either:

            - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
            - a path to a `directory` containing model weights saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
            - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

        model_args: (`optional`) Sequence of positional arguments:
            All remaning positional arguments will be passed to the underlying model's ``__init__`` method

        config: (`optional`) instance of a class derived from :class:`~pytorch_transformers.PretrainedConfig`:
            Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

            - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
            - the model was saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
            - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

        state_dict: (`optional`) dict:
            an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
            This option can be used if you want to create a model from a pretrained configuration but load your own weights.
            In this case though, you should check if using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and :func:`~pytorch_transformers.PreTrainedModel.from_pretrained` is not a simpler option.

        cache_dir: (`optional`) string:
            Path to a directory in which a downloaded pre-trained model
            configuration should be cached if the standard cache should not be used.

        force_download: (`optional`) boolean, default False:
            Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

        proxies: (`optional`) dict, default None:
            A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
            The proxies are used on each request.

        output_loading_info: (`optional`) boolean:
            Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

        kwargs: (`optional`) Remaining dictionary of keyword arguments:
            Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

            - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
            - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~pytorch_transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

    Examples::

        model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
        model = AutoModelForQuestionAnswering.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
        assert model.config.output_attention == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
        model = AutoModelForQuestionAnswering.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    """
    return AutoModelForQuestionAnswering.from_pretrained(*args, **kwargs)
