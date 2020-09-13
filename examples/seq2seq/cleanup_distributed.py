from pathlib import Path

import fire


try:
    from .utils import calculate_bleu, calculate_rouge, load_json, save_json, write_txt_file
except ImportError:
    from utils import calculate_bleu, calculate_rouge, load_json, save_json, write_txt_file


def cleanup_distributed_results(
    result_dir: str, save_dir: str = None, save_prefix=None, calc_bleu=False, just_metrics=False
):
    """Write first n lines of each file f in src_dir to dest_dir/f """
    src_dir = Path(result_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    records = sum([load_json(x) for x in src_dir.glob("*.json")])
    preds = [x["pred"] for x in records]
    labels = [x["label"] for x in records]
    score_fn = calculate_bleu if calc_bleu else calculate_rouge
    metrics = score_fn(preds, labels)
    save_json(metrics, save_dir.joinpath("metrics.json"))  # better would be be {prefix}_{rouge|bleu}.json
    if just_metrics:
        return

    if save_prefix is None:
        save_prefix = "generated"
        print("using generated as prefix")

    tgt_path = save_dir.joinpath(save_prefix, ".target")
    write_txt_file(labels, tgt_path)
    pred_path = save_dir.joinpath(save_prefix, ".pred_target")
    write_txt_file(preds, pred_path)
    if "source" in records[0]:
        src_path = save_dir.joinpath(save_prefix, ".source")
        write_txt_file([x["source"] for x in records], src_path)


if __name__ == "__main__":
    fire.Fire(cleanup_distributed_results)
