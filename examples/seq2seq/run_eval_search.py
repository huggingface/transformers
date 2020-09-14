import argparse
import itertools
import sys

from run_eval import run_generate


def parse_search_arg(search):
    groups = search.split()
    entries = {k: vs for k, vs in (g.split("=") for g in groups)}
    entry_names = list(entries.keys())
    sets = [list((f"--{k} {v}") for v in vs.split(":")) for k, vs in entries.items()]
    matrix = [list(x) for x in itertools.product(*sets)]
    return matrix, entry_names


def run_search():
    """
    Run parametric search over the desired hparam space with help of ``run_eval.py``.

    All the arguments except ``--search`` are passed to ``run_eval.py`` as is. The values inside of "--search" are parsed, reformatted and fed to ``run_eval.py`` as additional args.

   The format for the ``--search`` value is a simple string with hparams and colon separated values to try, e.g.:
   ```
    --search "num_beams=5:10 length_penalty=0.8:1.0:1.2 early_stopping=true:false"
   ```
   which will generate ``12`` ``(2*3*2)`` searches for a product of each hparam. For example the example that was just used will invoke ``run_eval.py`` repeatedly with:
   
   ```
    --num_beams 5 --length_penalty 0.8 --early_stopping true
    --num_beams 5 --length_penalty 0.8 --early_stopping false
    [...]
    --num_beams 10 --length_penalty 1.2 --early_stopping false
   ```
   
   On completion, this function prints a markdown table of the results sorted by the best BLEU score and the winning arguments.


    """
    prog = sys.argv[0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search",
        type=str,
        required=False,
        help='param space to search, e.g. "num_beams=5:10 length_penalty=0.8:1.0:1.2"',
    )

    args, args_main = parser.parse_known_args()
    args_normal = [prog] + args_main

    matrix, col_names = parse_search_arg(args.search)
    col_names.insert(0, "bleu")
    col_widths = {col: len(str(col)) for col in col_names}
    results = []
    for r in matrix:
        args_exp = " ".join(r).split()
        sys.argv = args_normal + args_exp
        scores, hparams = run_generate(verbose=False)
        result = hparams
        result["bleu"] = f"{scores['bleu']:0.2f}"
        results.append(result)

        # find widest entries
        for k, v in result.items():
            l = len(str(v))
            if l > col_widths[k]:
                col_widths[k] = l

    results_sorted = sorted(results, key=lambda x: x["bleu"], reverse=True)
    print(" | ".join([f"{col:{col_widths[col]}}" for col in col_names]))
    print(" | ".join([f"{'-'*col_widths[col]}" for col in col_names]))
    for row in results_sorted:
        print(" | ".join([f"{row[col]:{col_widths[col]}}" for col in col_names]))

    best = results_sorted[0]
    del best["bleu"]
    best_args = [f"--{k} {v}" for k, v in best.items()]
    print("\nBest score args:")
    print(" ".join(args_main + best_args))

    return results_sorted


if __name__ == "__main__":
    # Usage:
    # [normal-run_eval_search.py cmd plus] \
    # --search="num_beams=1:5:10 length_penalty=0.8:1:1.2 early_stopping=true:false"
    #
    # Example:
    # PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval_search.py $MODEL_NAME \
    # $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target \
    # --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation \
    # --search="num_beams=1:5:10 length_penalty=0.8:1:1.2 early_stopping=true:false"
    run_search()
