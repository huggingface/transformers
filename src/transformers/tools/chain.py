class SimpleChainer:
    """
    Example:

    ```py
    from transformers import pipeline
    from transformers.tools import SimpleChainer

    chain = SimpleChainer(
        pipeline("translation", model="facebook/nllb-200-distilled-600M", src_lang="fra_Latn", tgt_lang="eng_Latn"),
        pipeline(model="gpt2"),
    )
    chain("Je m'appelle Lysandre et mon bagel préféré est")
    ```
    """

    def __init__(self, *functions):
        self.functions = functions

    def __call__(self, *args, **kwargs):
        output = self.functions[0](*args, **kwargs)
        for func in self.functions[1:]:
            if isinstance(output, (tuple, list)):
                if len(output) == 1 and isinstance(output[0], dict):
                    if len(output[0]) == 1:
                        output = func(*list(output[0].values()))
                    else:
                        output = func(**output[0])
                else:
                    output = func(*output[0])
            elif isinstance(output, dict):
                if len(output) == 1:
                    output = func(*list(output.values()))
                else:
                    output = func(**output)
            else:
                output = func(output)
        return output
