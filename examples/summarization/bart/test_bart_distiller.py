import unittest

from durbango import DEFAULT_DEVICE
from textbrewer import TrainingConfig, DistillationConfig, GeneralDistiller
from torch.utils.data import DataLoader
from transformers import AdamW, BartTokenizer
from transformers.file_utils import cached_property

from .bart_distiller import make_teacher_and_student, simple_adaptor
from .utils import SummarizationDataset
from pathlib import Path

class TestDistiller(unittest.TestCase):

    @cached_property
    def tok(self):
        return BartTokenizer.from_pretrained('bart-large')

    @property
    def loader(self):
        DATA_DIR = '/Users/shleifer/Dropbox/cnn_tiny/'
        dataset = SummarizationDataset(self.tok,
                                       data_dir=DATA_DIR, max_source_length=12, max_target_length=6)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn, shuffle=False)
        return dataloader

    def test_bdistiller_tiny(self):
        teacher_cfg_kwargs = dict(output_hidden_states=True, encoder_attention_heads=2, decoder_attention_heads=2,
                                  encoder_ffn_dim=128, decoder_ffn_dim=32, encoder_layers=3, decoder_layers=3,
                                  d_model=128)
        Path('distil_tiny_log_dir').mkdir(exist_ok=True)
        teacher_model, student_model = make_teacher_and_student(teacher_cfg_kwargs, d_model=64)
        train_config = TrainingConfig(device=DEFAULT_DEVICE, log_dir='distil_tiny_log_dir')
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

        dataloader = self.loader
        with distiller:
            opt = AdamW(student_model.parameters(), lr=.001, )
            start_loss = distiller.train(opt, dataloader, num_epochs=1,
                            callback=None)
        with distiller:
            opt = AdamW(student_model.parameters(), lr=.1, )
            end_loss = distiller.train(opt, dataloader, num_epochs=10,
                            callback=None)
        self.assertGreaterEqual(start_loss, end_loss)
        #print(f'end_loss: {end_loss: .4f}, start_loss: {start_loss: .4f}')
