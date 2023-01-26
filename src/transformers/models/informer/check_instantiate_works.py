from transformers import InformerModel, InformerConfig, TimeSeriesTransformerModel
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str

from huggingface_hub import hf_hub_download
import torch


if __name__ == '__main__':
    freq = "1M"
    prediction_length = 24
    lags = get_lags_for_frequency(freq_str=freq)
    time_features = time_features_from_frequency_str(freq)

    config = InformerConfig(prediction_length=prediction_length,
                            context_length=prediction_length*3,
                            lags_seq=lags,
                            num_time_features=len(time_features) + 1,
                            num_static_categorical_features=1,
                            cardinality=[366],
                            embedding_dimension=[2],
                            encoder_layers=4,
                            decoder_layers=4)

    model: InformerModel = InformerModel(config)
    print(model)
    file = hf_hub_download(
        repo_id="kashif/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
    )
    batch = torch.load(file)

    # during training, one provides both past and future values
    # as well as possible additional features
    outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"],
        static_real_features=batch["static_real_features"],
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
    )

    last_hidden_state = outputs.last_hidden_state
