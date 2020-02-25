import typer


def download(
    model: str,
    cache_dir: str = typer.Option(None, help="Path to location to store the models"),
    force: bool = typer.Option(False, help="Force the model to be download even if already in cache-dir"),
):
    """Download a pretrained model.
    
    Example::

        $ transformers download bert-base-uncased
    """

    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(model, cache_dir=cache_dir, force_download=force)
    AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, force_download=force)
