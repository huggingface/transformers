import json
import argparse
from utlis_nq_eval import EVAL_OPTS_NQ, main as evaluate_on_nq

if __name__ == '__main__':
    #--------------------------------------------------------------
    all_prediction_files = []
    output_prediction_file = ""
    gold_gzip_path = "/mnt/NQ_Data/dev"
    # --------------------------------------------------------------
    all_preds = []
    for per_pred_file in all_prediction_files:
        with open(per_pred_file, "r") as fin:
            per_results = json.load(fin)
        all_preds.append(per_results["predictions"])
    predictions_json = {"predictions": all_preds}
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(predictions_json, indent=4) + "\n")
    print("dumped all predictions to {}".format(output_prediction_file))
    # --------------------------------------------------------------
    evaluate_options = EVAL_OPTS_NQ(gold_path = gold_gzip_path,
                                    pred_path = output_prediction_file)
    results = evaluate_on_nq(evaluate_options)
    print(json.dumps(results, indent=2))