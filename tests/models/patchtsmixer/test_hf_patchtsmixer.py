import unittest
import torch
from transformers.models.patchtsmixer.configuration_patchtsmixer import PatchTSMixerConfig
from transformers.models.patchtsmixer.modeling_patchtsmixer import (
    PatchTSMixerEncoder,
    PatchTSMixerModel,
    PatchTSMixerPretrainHead,
    PatchTSMixerForPretraining,
    PatchTSMixerForecastHead,
    PatchTSMixerForForecasting,
    PatchTSMixerClassificationHead,
    PatchTSMixerForClassification,
    PatchTSMixerRegressionHead,
    PatchTSMixerForRegression,
)


class TestHFPatchTSMixer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup method: Called once before test-cases execution"""
        cls.params = {}
        cls.params.update(
            seq_len=32,
            patch_len=8,
            in_channels=3,
            stride=8,
            num_features=4,
            expansion_factor=2,
            num_layers=3,
            dropout=0.2,
            mode="common_channel",  # common_channel, flatten, mix_channel
            gated_attn=True,
            norm_mlp="LayerNorm",
            mask_type="random",
            mask_ratio=0.5,
            mask_patches=[2, 3],
            mask_patch_ratios=[1, 1],
            mask_value=0,
            masked_loss = True,
            channel_consistent_masking=True,
            head_dropout=0.2,
            forecast_len=64,
            out_channels=None,
            n_classes=3,
            n_targets=3,
            output_range=None,
            head_agg=None,
            revin = True,
        )

        cls.num_patches = (
            max(cls.params["seq_len"], cls.params["patch_len"])
            - cls.params["patch_len"]
        ) // cls.params["stride"] + 1

        # batch_size = 32
        batch_size = 2

        out_patches = int(cls.params["forecast_len"] / cls.params["patch_len"])

        cls.data = torch.rand(
            batch_size,
            cls.params["seq_len"],
            cls.params["in_channels"],
        )

        cls.enc_data = torch.rand(
            batch_size,
            cls.params["in_channels"],
            cls.num_patches,
            cls.params["patch_len"],
        )

        cls.enc_output = torch.rand(
            batch_size,
            cls.params["in_channels"],
            cls.num_patches,
            cls.params["num_features"],
        )

        cls.correct_pred_output = torch.rand(
            batch_size, cls.params["forecast_len"], cls.params["in_channels"]
        )
        cls.correct_regression_output = torch.rand(batch_size, cls.params["n_targets"])

        cls.correct_pretrain_output = torch.rand(
            batch_size,
            cls.params["in_channels"],
            cls.num_patches,
            cls.params["patch_len"],
        )

        cls.correct_forecast_output = torch.rand(
            batch_size,
            cls.params["forecast_len"],
            cls.params["in_channels"],
        )

        cls.correct_sel_forecast_output = torch.rand(
            batch_size,
            cls.params["forecast_len"],
            2
        )

        cls.correct_classification_output = torch.rand(
            batch_size,
            cls.params["n_classes"],
        )

        cls.correct_classification_classes = torch.randint(
            0, cls.params["n_classes"], (batch_size,)
        )

    
    def test_patchtsmixer_encoder(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        enc = PatchTSMixerEncoder(config)
        output = enc(self.__class__.enc_data)
        self.assertEqual(output.shape, self.__class__.enc_output.shape)

    def test_patchmodel(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerModel(config)
        output = mdl(self.__class__.data)
        self.assertEqual(
            output.last_hidden_state.shape, self.__class__.enc_output.shape
        )
        self.assertEqual(output.last_hidden_state.shape, output[0].shape)
        self.assertEqual(
            output.patched_input.shape, self.__class__.enc_data.shape
        )

    def test_pretrainhead(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerPretrainHead(config)
        output = head(self.__class__.enc_output)
        
        self.assertEqual(output.shape, self.__class__.correct_pretrain_output.shape)

    def test_pretrain_full(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForPretraining(config)
        output = mdl(self.__class__.data)
        self.assertEqual(
            output.prediction_logits.shape, self.__class__.correct_pretrain_output.shape
        )
        self.assertEqual(
            output.backbone_embeddings.shape, self.__class__.enc_output.shape
        )
        self.assertEqual(output.loss.item()<100,True)

        # print("loss shape", output.loss, output.loss.shape)

    def test_forecast_head(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerForecastHead(config)
        # output = head(self.__class__.enc_output, raw_data = self.__class__.correct_pretrain_output)
        output = head(self.__class__.enc_output)
        
        self.assertEqual(output.shape, self.__class__.correct_forecast_output.shape)

    def forecast_full_module(self, params):
        config = PatchTSMixerConfig(**params)
        mdl = PatchTSMixerForForecasting(config)

        target_val = self.__class__.correct_forecast_output

        if config.forecast_channel_indices is not None:
            target_val = self.__class__.correct_sel_forecast_output


        output = mdl(self.__class__.data, target_values=target_val)

        self.assertEqual(
                output.prediction_logits.shape, target_val.shape
            )

        self.assertEqual(
            output.backbone_embeddings.shape, self.__class__.enc_output.shape
        )
        self.assertEqual(output.loss.item()<100,True)
        # print("loss shape", output.loss, output.loss.shape)

    def test_forecast_full(self):
        self.forecast_full_module(self.__class__.params)
    
    def test_forecast_full_2(self):
        params = self.__class__.params.copy()
        params.update(mode = "mix_channel",
                        swin_hier = 2,
                        decoder_mode = "mix_channel", 
                        )
        self.forecast_full_module(params)

    def test_forecast_full_3(self):
        params = self.__class__.params.copy()
        params.update(mode = "mix_channel",
                        swin_hier = 2,
                        decoder_mode = "mix_channel", 
                        )
        self.forecast_full_module(params)
    
    def test_forecast_full_3(self):
        params = self.__class__.params.copy()
        params.update(mode = "mix_channel",
                        swin_hier = 2,
                        decoder_mode = "mix_channel", 
                        forecast_channel_indices = [0,2],
                        )
        self.forecast_full_module(params)


    # def test_forecast_full(self):
    #     config = PatchTSMixerConfig(**self.__class__.params)
    #     mdl = PatchTSMixerForForecasting(config)
    #     output = mdl(self.__class__.data, target_values=self.__class__.correct_forecast_output)
    #     self.assertEqual(
    #         output.prediction_logits.shape, self.__class__.correct_forecast_output.shape
    #     )
    #     self.assertEqual(
    #         output.backbone_embeddings.shape, self.__class__.enc_output.shape
    #     )
    #     self.assertEqual(output.loss.item()<100,True)
    #     # print("loss shape", output.loss, output.loss.shape)

    def test_classification_head(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerClassificationHead(config)
        # output = head(self.__class__.enc_output, raw_data = self.__class__.correct_pretrain_output)
        output = head(self.__class__.enc_output)
        
        self.assertEqual(
            output.shape, self.__class__.correct_classification_output.shape
        )

    def test_classification_full(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForClassification(config)
        output = mdl(
            self.__class__.data, target_values=self.__class__.correct_classification_classes
        )
        self.assertEqual(
            output.prediction_logits.shape,
            self.__class__.correct_classification_output.shape,
        )
        self.assertEqual(
            output.backbone_embeddings.shape, self.__class__.enc_output.shape
        )
        self.assertEqual(output.loss.item()<100,True)
        # print("loss shape", output.loss, output.loss.shape)

    def test_regression_head(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerRegressionHead(config)
        # output = head(self.__class__.enc_output, raw_data = self.__class__.correct_pretrain_output)
        output = head(self.__class__.enc_output)
        # print(output.shape)
        self.assertEqual(output.shape, self.__class__.correct_regression_output.shape)

    def test_regression_full(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        mdl = PatchTSMixerForRegression(config)
        output = mdl(self.__class__.data, target_values=self.__class__.correct_regression_output)
        self.assertEqual(
            output.prediction_logits.shape,
            self.__class__.correct_regression_output.shape,
        )
        self.assertEqual(
            output.backbone_embeddings.shape, self.__class__.enc_output.shape
        )
        self.assertEqual(output.loss.item()<100,True)

        # print("loss shape", output.loss, output.loss.shape)
