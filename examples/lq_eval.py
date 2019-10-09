import json
import argparse
from utlis_nq_eval import EVAL_OPTS_NQ, main as evaluate_on_nq
evaluate_options = EVAL_OPTS_NQ(gold_path ="/mnt/NQ_Data/dev" ,
                                pred_path="/mnt/gcrnqdata/model/bert-large-384-squadft-0.02nsp/predictions_.json")
results = evaluate_on_nq(evaluate_options)
print(json.dumps(results, indent=2))