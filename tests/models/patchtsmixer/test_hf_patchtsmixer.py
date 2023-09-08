import unittest
import torch
from transformers.models.patchtsmixer.configuration_patchtsmixer import PatchTSMixerConfig
from transformers.models.patchtsmixer.modeling_patchtsmixer import (
    PatchTSMixerEncoder,
    PatchTSMixerModel,
    PatchTSMixerPretrainHead,
    PatchTSMixerForMaskPretraining,
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
            use_pe = False,
            pe = "sincos",
            learn_pe = True,
            self_attn = False,
            self_attn_heads = 1,
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

        cls.flat_enc_output = torch.rand(
            batch_size,
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
        self.assertEqual(output.last_hidden_state.shape, self.__class__.enc_output.shape)

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
        mdl = PatchTSMixerForMaskPretraining(config)
        output = mdl(self.__class__.data)
        self.assertEqual(
            output.prediction_logits.shape, self.__class__.correct_pretrain_output.shape
        )
        self.assertEqual(
            output.last_hidden_state.shape, self.__class__.enc_output.shape
        )
        self.assertEqual(output.loss.item()<100,True)

        # print("loss shape", output.loss, output.loss.shape)

    def test_forecast_head(self):
        config = PatchTSMixerConfig(**self.__class__.params)
        head = PatchTSMixerForecastHead(config)
        # output = head(self.__class__.enc_output, raw_data = self.__class__.correct_pretrain_output)
        output = head(self.__class__.enc_output)
        
        self.assertEqual(output.shape, self.__class__.correct_forecast_output.shape)

    def check_module(self, task, params = None, output_hidden_states = True,):
        config = PatchTSMixerConfig(**params)
        if task == "forecast":
            mdl = PatchTSMixerForForecasting(config)
            target_input = self.__class__.correct_forecast_output
            if config.forecast_channel_indices is not None:
                target_output = self.__class__.correct_sel_forecast_output
            else:
                target_output = target_input
        
        elif task == "classification":
            mdl = PatchTSMixerForClassification(config)
            target_input = self.__class__.correct_classification_classes
            target_output = self.__class__.correct_classification_output
        elif task == "regression":
            mdl = PatchTSMixerForRegression(config)
            target_input = self.__class__.correct_regression_output
            target_output = self.__class__.correct_regression_output
        elif task == "pretrain":
            mdl = PatchTSMixerForMaskPretraining(config)
            target_input = None
            target_output = self.__class__.correct_pretrain_output
        else:
            print("invalid task")
        
        if config.mode == "flatten":
            enc_output = self.__class__.flat_enc_output
        else:
            enc_output = self.__class__.enc_output
        
        if target_input is None:
            output = mdl(self.__class__.data, 
                    output_hidden_states = output_hidden_states)
        else:
            output = mdl(self.__class__.data, 
                    target_values=target_input, 
                    output_hidden_states = output_hidden_states)

        self.assertEqual(
                output.prediction_logits.shape, target_output.shape
            )

        self.assertEqual(
            output.last_hidden_state.shape, enc_output.shape
        )

        if output_hidden_states is True:
            self.assertEqual(len(output.hidden_states),params["num_layers"])
            
        else:
            self.assertEqual(output.hidden_states, None)
            
        
        self.assertEqual(output.loss.item()<100,True)


    def test_forecast(self):
        for mode in ["flatten","common_channel","mix_channel"]:
            for self_attn in [True,False]:
                for revin in [True, False]:
                    for gated_attn in [True, False]:
                        for forecast_channel_indices in [None, [0,2]]:
                            params = self.__class__.params.copy()
                            params.update(mode = mode, 
                                self_attn=self_attn,
                                revin=revin,
                                forecast_channel_indices=forecast_channel_indices,
                                gated_attn = gated_attn)
                            
                            self.check_module(task="forecast",params=params)

    def test_classification(self):
        for mode in ["common_channel","mix_channel","flatten"]:
            for self_attn in [True,False]:
                for revin in [True, False]:
                    for gated_attn in [True, False]:
                        for head_agg in ["max_pool","avg_pool"]:
                            if mode == "flatten" and revin is True:
                                continue
                            params = self.__class__.params.copy()
                            params.update(mode = mode, 
                                self_attn=self_attn,
                                revin=revin,
                                head_agg=head_agg,
                                gated_attn = gated_attn)
                            # print(mode,self_attn,revin,gated_attn,head_agg)
                        
                            self.check_module(task="classification",params=params)

    def test_regression(self):
        for mode in ["common_channel","mix_channel","flatten"]:
            for self_attn in [True,False]:
                for revin in [True, False]:
                    for gated_attn in [True, False]:
                        for head_agg in ["max_pool","avg_pool"]:
                            if mode == "flatten" and revin is True:
                                continue
                            params = self.__class__.params.copy()
                            params.update(mode = mode, 
                                self_attn=self_attn,
                                revin=revin,
                                head_agg=head_agg,
                                gated_attn = gated_attn)
                            # print(mode,self_attn,revin,gated_attn,head_agg)
                        
                            self.check_module(task="regression",params=params)


    
    def test_pretrain(self):
        for mode in ["common_channel","mix_channel","flatten"]:
            for self_attn in [True,False]:
                for revin in [True, False]:
                    for gated_attn in [True, False]:
                        for mask_type in ["random","forecast"]:
                            for masked_loss in [True, False]:
                                for channel_consistent_masking in [True, False]:
                                    params = self.__class__.params.copy()
                                    params.update(mode = mode, 
                                        self_attn=self_attn,
                                        revin=revin,
                                        gated_attn = gated_attn,
                                        mask_type = mask_type,
                                        masked_loss = masked_loss,
                                        channel_consistent_masking = channel_consistent_masking,
                                        )
                                    # print(mode,self_attn,revin,gated_attn,head_agg)
                                
                                    self.check_module(task="pretrain",params=params)

        # for mode in ["flatten","common_channel","mix_channel"]:
        #     for task in ["forecast","classification","regression","pretrain"]:
        #         for self_attn in [True,False]:
        #             for head_agg in ["max_pool","avg_pool"]:
        #                 for mask_type in ["random","forecast"]:
        #                     for masked_loss in [True, False]:
        #                         for channel_consistent_masking in [True, False]:
        #                             for revin in [True, False]:
        #                                 for forecast_channel_indices in [None, [0,2]]:








        

        

        

    def forecast_full_module(self, params = None, output_hidden_states = False):
        
        config = PatchTSMixerConfig(**params)
        mdl = PatchTSMixerForForecasting(config)

        target_val = self.__class__.correct_forecast_output

        if config.forecast_channel_indices is not None:
            target_val = self.__class__.correct_sel_forecast_output

        if config.mode == "flatten":
            enc_output = self.__class__.flat_enc_output
        else:
            enc_output = self.__class__.enc_output
        
        output = mdl(self.__class__.data, target_values=self.__class__.correct_forecast_output, output_hidden_states = output_hidden_states)

        self.assertEqual(
                output.prediction_logits.shape, target_val.shape
            )

        self.assertEqual(
            output.last_hidden_state.shape, enc_output.shape
        )

        if output_hidden_states is True:
            self.assertEqual(len(output.hidden_states),params["num_layers"])
            
        else:
            self.assertEqual(output.hidden_states, None)
            
        
        self.assertEqual(output.loss.item()<100,True)
        # print("loss shape", output.loss, output.loss.shape)

    def test_forecast_full(self):
        self.check_module(task = "forecast",
                            params = self.__class__.params,
                            output_hidden_states = True
                            )
        # self.forecast_full_module(self.__class__.params, output_hidden_states = True)
    
    def test_forecast_full_2(self):
        params = self.__class__.params.copy()
        params.update(mode = "mix_channel",
                        )
        self.forecast_full_module(params,  output_hidden_states = True)

    def test_forecast_full_3(self):
        params = self.__class__.params.copy()
        params.update(mode = "flatten",
                        )
        self.forecast_full_module(params, output_hidden_states = True)
    
    def test_forecast_full_5(self):
        params = self.__class__.params.copy()
        params.update(self_attn = True,
                        use_pe = True,
                        pe = "sincos",
                        learn_pe = True,
                        )
        self.forecast_full_module(params, output_hidden_states = True)

    def test_forecast_full_4(self):
        params = self.__class__.params.copy()
        params.update(mode = "mix_channel",
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
    #         output.last_hidden_state.shape, self.__class__.enc_output.shape
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
            output.last_hidden_state.shape, self.__class__.enc_output.shape
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
            output.last_hidden_state.shape, self.__class__.enc_output.shape
        )
        self.assertEqual(output.loss.item()<100,True)

        # print("loss shape", output.loss, output.loss.shape)
