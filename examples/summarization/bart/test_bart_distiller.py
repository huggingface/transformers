import unittest

from durbango import DEFAULT_DEVICE
from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller
from torch.utils.data import DataLoader
from transformers import AdamW, BartTokenizer

from .bart_distiller import make_teacher_and_student, simple_adaptor
from .utils import SummarizationDataset


class TestDistiller(unittest.TestCase):

    def test_bdistiller_tiny(self):
        teacher_cfg_kwargs = dict(output_hidden_states=True, encoder_attention_heads=2, decoder_attention_heads=2,
                                  encoder_ffn_dim=128, decoder_ffn_dim=32, encoder_layers=3, decoder_layers=3,
                                  d_model=128)

        teacher_model, student_model = make_teacher_and_student(teacher_cfg_kwargs, d_model=64)
        train_config = TrainingConfig(device=DEFAULT_DEVICE)
        # Matching different layers of the student and the teacher
        distill_config = DistillationConfig(
            intermediate_matches=[
                # {'layer_T':0, 'layer_S':0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
                # {'layer_T':8, 'layer_S':2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1}
            ])

        # Build distiller
        distiller = GeneralDistiller(
            train_config=train_config, distill_config=distill_config,
            model_T=teacher_model, model_S=student_model,
            adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)
        opt = AdamW(student_model.parameters())
        DATA_DIR = '/Users/shleifer/Dropbox/cnn_tiny/'
        tok = BartTokenizer.from_pretrained('bart-large')

        dataset = SummarizationDataset(tok,
                                       data_dir=DATA_DIR, max_source_length=12, max_target_length=6)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn, shuffle=False)
        with distiller:
            distiller.train(opt, dataloader, num_epochs=4,
                            callback=None)
