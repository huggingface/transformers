from transformers import InformerModel, InformerConfig
from gluonts.time_feature import get_lags_for_frequency

if __name__ == '__main__':
    freq = "h"
    lags = get_lags_for_frequency(freq_str=freq)
    model = InformerModel(InformerConfig())
    print(model)
