from train_ft_eli5 import *

from nltk import PorterStemmer
from rouge import Rouge
from spacy.lang.en import English
from time import time

stemmer = PorterStemmer()
rouge = Rouge()
tokenizer = English().Defaults.create_tokenizer()


def compute_rouge(compare_list):
    preds = [" ".join([stemmer.stem(str(w))
                       for w in tokenizer(pred)])
             for gold, pred in compare_list]
    golds = [" ".join([stemmer.stem(str(w))
                       for w in tokenizer(gold)])
             for gold, pred in compare_list]
    scores = rouge.get_scores(preds, golds, avg=True)
    return scores

# BART wiki perplexity: 17.620
# ROUGE: 25.63 / 5.82 / 26.50

def main():
    parser = argparse.ArgumentParser()
    # optimization arguments
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=2,
        type=int,
        help="Number of examples per batch (on GPU)",
    )
    parser.add_argument(
        "-edf",
        "--evaluation_data",
        default="explainlikeimfive_valid.json",
        type=str,
        help="evaluation data",
    )
    parser.add_argument(
        "-eds",
        "--evaluation_data_size",
        default=-1,
        type=int,
        help="truncate size of evaluation dataset",
    )
    parser.add_argument(
        "--max_length",
        default=1024,
        type=int,
        help="Maximum sequence length for both input and output",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        default="eli5_model",
        type=str,
        help="name of saved model",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        default=15,
        type=int,
        help="Training epoch to load parameters for",
    )
    # generation arguments
    parser.add_argument(
        "-mil",
        "--min_generation_length",
        default=128,
        type=int,
        help="Min length for generation",
    )
    parser.add_argument(
        "-mal",
        "--max_generation_length",
        default=512,
        type=int,
        help="Maximum length for generation",
    )
    parser.add_argument(
        "-rp",
        "--rep_penalty",
        default=2.,
        type=float,
        help="Repetition penalty",
    )
    parser.add_argument(
        "-pf",
        "--print_freq",
        default=100,
        type=int,
        help="Printing frequency",
    )
    args_run = parser.parse_args()
    print("---- Loading model")
    model, tokenizer, args = load_saved(
        '{}_args.pk'.format(args_run.model_name),
        '{}_{}.pth'.format(args_run.model_name, args_run.epoch),
    )
    args.max_length = args_run.max_length
    args.batch_size = args_run.batch_size
    _ = model.eval()
    _ = model.to('cuda:0')
    print("---- Loading evaluation data")
    eval_data = json.load(open(args_run.evaluation_data, encoding='utf-8'))
    if args_run.evaluation_data_size > 0:
        eval_data = eval_data[:args_run.evaluation_data_size]
    eval_dset = ELI5Dataset(eval_data, tokenizer, args)
    print("---- Computing perplexity on evaluation data")
    eval_loss = evaluate(model, eval_dset, tokenizer, args, args_run.print_freq)
    print("---- Evaluation loss: {:.3f} \t perplexity: {:.3f}".format(eval_loss, math.exp(eval_loss)))
    compare = []
    st_time = time()
    for i, qda_dct in enumerate(eval_data):
        pred = answer_example(
            qda_dct, model, tokenizer, args,
            min_length=args_run.min_generation_length,
            max_length=args_run.max_generation_length,
            rep_pen=args_run.rep_penalty,
            n_beams=args_run.batch_size,
            n_samples=0, verbose=False
        )['beam']
        compare += [(qda_dct['id'], qda_dct['question'], (qda_dct['answer'], pred))]
        if i % 10 == 0:
            print(i, time() - st_time)
            scores = compute_rouge([x[-1] for x in compare])
            print('R-1: {:.2f} \t R-2: {:.2f} \t R-L: {:.2f}'.format(
                scores['rouge-1']['f'] * 100,
                scores['rouge-2']['f'] * 100,
                scores['rouge-l']['f'] * 100))
    print('R-1: {:.2f} \t R-2: {:.2f} \t R-L: {:.2f}'.format(
                scores['rouge-1']['f'] * 100,
                scores['rouge-2']['f'] * 100,
                scores['rouge-l']['f'] * 100))
    
    

if __name__ == "__main__":
    main()

