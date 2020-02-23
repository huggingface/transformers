import typer


def download(
    model: str,
    test_n: int,
    cache_dir: str = typer.Option(None, help="Path to location to store the models"),
    force: bool = typer.Option(False, help="Force the model to be download even if already in cache-dir"),
):
    """CLI tool to download a model."""

    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(model, cache_dir=cache_dir, force_download=force)
    AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, force_download=force)
