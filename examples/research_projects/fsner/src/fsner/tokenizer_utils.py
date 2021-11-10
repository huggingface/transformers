import torch

from transformers import AutoTokenizer


class FSNERTokenizerUtils(object):
    def __init__(self, pretrained_model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def tokenize(self, x):
        """
        Wrapper function for tokenizing query and supports
        Args:
            x (`List[str] or List[List[str]]`):
                List of strings for query or list of lists of strings for supports.
        Returns:
            `transformers.tokenization_utils_base.BatchEncoding` dict with additional keys and values for start_token_id, end_token_id and sizes of example lists for each entity type
        """

        if isinstance(x, list) and all([isinstance(_x, list) for _x in x]):
            d = None
            for l in x:
                t = self.tokenizer(
                    l,
                    padding="max_length",
                    max_length=384,
                    truncation=True,
                    return_tensors="pt",
                )
                t["sizes"] = torch.tensor([len(l)])
                if d is not None:
                    for k in d.keys():
                        d[k] = torch.cat((d[k], t[k]), 0)
                else:
                    d = t

            d["start_token_id"] = torch.tensor(self.tokenizer.convert_tokens_to_ids("[E]"))
            d["end_token_id"] = torch.tensor(self.tokenizer.convert_tokens_to_ids("[/E]"))

        elif isinstance(x, list) and all([isinstance(_x, str) for _x in x]):
            d = self.tokenizer(
                x,
                padding="max_length",
                max_length=384,
                truncation=True,
                return_tensors="pt",
            )

        else:
            raise Exception(
                "Type of parameter x was not recognized! Only `list of strings` for query or `list of lists of strings` for supports are supported."
            )

        return d

    def extract_entity_from_scores(self, query, W_query, p_start, p_end, thresh=0.70):
        """
        Extracts entities from query and scores given a threshold.
        Args:
            query (`List[str]`):
                List of query strings.
            W_query (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of query sequence tokens in the vocabulary.
            p_start (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Scores of each token as being start token of an entity
            p_end (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Scores of each token as being end token of an entity
            thresh (`float`):
                Score threshold value
        Returns:
            A list of lists of tuples(decoded entity, score)
        """

        final_outputs = []
        for idx in range(len(W_query["input_ids"])):
            start_indexes = end_indexes = range(p_start.shape[1])

            output = []
            for start_id in start_indexes:
                for end_id in end_indexes:
                    if start_id < end_id:
                        output.append(
                            (
                                start_id,
                                end_id,
                                p_start[idx][start_id].item(),
                                p_end[idx][end_id].item(),
                            )
                        )

            output.sort(key=lambda tup: (tup[2] * tup[3]), reverse=True)
            temp = []
            for k in range(len(output)):
                if output[k][2] * output[k][3] >= thresh:
                    c_start_pos, c_end_pos = output[k][0], output[k][1]
                    decoded = self.tokenizer.decode(W_query["input_ids"][idx][c_start_pos:c_end_pos])
                    temp.append((decoded, output[k][2] * output[k][3]))

            final_outputs.append(temp)

        return final_outputs
