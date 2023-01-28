from transformers import InformerModel, InformerConfig, TimeSeriesTransformerForPrediction, TimeSeriesTransformerModel, \
    TimeSeriesTransformerConfig
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str

from huggingface_hub import hf_hub_download
import torch

"""
Establish one batch for forward pass in the Informer
"""
if __name__ == '__main__':
    freq = "1M"
    prediction_length = 24
    lags = get_lags_for_frequency(freq_str=freq)
    time_features = time_features_from_frequency_str(freq)

    config = InformerConfig(prediction_length=prediction_length,
                            context_length=prediction_length*3,
                            lags_sequence=lags,
                            num_time_features=len(time_features) + 1,
                            num_static_categorical_features=1,
                            cardinality=[366],
                            embedding_dimension=[2],
                            encoder_layers=1,
                            decoder_layers=1)
    model = InformerModel(config)
    print(model)

    # config = TimeSeriesTransformerConfig(
    #     prediction_length=prediction_length,
    #     context_length=prediction_length * 3,  # context length
    #     lags_sequence=lags,
    #     num_time_features=len(time_features) + 1,  # we'll add 2 time features ("month of year" and "age", see further)
    #     num_static_categorical_features=1,  # we have a single static categorical feature, namely time series ID
    #     cardinality=[366],  # it has 366 possible values
    #     embedding_dimension=[2],  # the model will learn an embedding of size 2 for each of the 366 possible values
    #     encoder_layers=4,
    #     decoder_layers=4,
    # )
    # model = TimeSeriesTransformerModel(config)
    # model.eval()
    #
    # model = TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")
    #
    # file = hf_hub_download(
    #     repo_id="kashif/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
    # )
    # batch = torch.load(file)
    #
    # # during training, one provides both past and future values
    # # as well as possible additional features
    # outputs = model(
    #     past_values=batch["past_values"],
    #     past_time_features=batch["past_time_features"],
    #     past_observed_mask=batch["past_observed_mask"],
    #     static_categorical_features=batch["static_categorical_features"],
    #     static_real_features=batch["static_real_features"],
    #     future_values=batch["future_values"],
    #     future_time_features=batch["future_time_features"],
    # )
    #
    # print(outputs.last_hidden_state.shape)
