from transformers import AutoTokenizer


class FSNERTokenizerUtils(object):
    def __init__(self, pretrained_model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def tokenize(self, x):
        return self.tokenizer(x, padding="max_length", max_length=384, truncation=True, return_tensors="pt")

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
                        output.append((start_id, end_id, p_start[idx][start_id].item(), p_end[idx][end_id].item()))

            output.sort(key=lambda tup: (tup[2] * tup[3]), reverse=True)
            temp = []
            for k in range(len(output)):
                if output[k][2] * output[k][3] >= thresh:
                    c_start_pos, c_end_pos = output[k][0], output[k][1]
                    decoded = self.tokenizer.decode(W_query["input_ids"][idx][c_start_pos:c_end_pos])
                    temp.append((decoded, output[k][2] * output[k][3]))

            final_outputs.append(temp)

        return final_outputs
