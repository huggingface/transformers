import json
import argparse
import os
from utlis_nq_eval import EVAL_OPTS_NQ, main as evaluate_on_nq

if __name__ == '__main__':
    #--------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_gzip_dir',required=True, default='', type=str)
    parser.add_argument('--input_prediction_dir', required=True,type=str)
    args = parser.parse_args()
    #---------input: splited pred files, output: jointed prediction file-------------
    all_prediction_files = []
    for i in range(4):
        all_prediction_files.append(os.join(args.input_prediction_dir,'predictions_{}_.json'.format(i)))
    output_prediction_file = os.join(args.input_prediction_dir,'all_predictions.json')
    gold_gzip_path = args.eval_gzip_dir
    # ----------------------------joint----------------------------------
    all_preds = []
    for per_pred_file in all_prediction_files:
        with open(per_pred_file, "r") as fin:
            per_results = json.load(fin)
        all_preds.append(per_results["predictions"])
    predictions_json = {"predictions": all_preds}
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(predictions_json, indent=4) + "\n")
    print("dumped all predictions to {}".format(output_prediction_file))
    # ----------------------------eval----------------------------------
    evaluate_options = EVAL_OPTS_NQ(gold_path = gold_gzip_path,
                                    pred_path = output_prediction_file)
    results = evaluate_on_nq(evaluate_options)
    print(json.dumps(results, indent=2))