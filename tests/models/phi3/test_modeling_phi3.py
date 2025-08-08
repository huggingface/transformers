# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Testing suite for the PyTorch Phi-3 model."""

import unittest

from transformers import Phi3Config, StaticCache, is_torch_available
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.testing_utils import (
    Expectations,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        Phi3ForCausalLM,
        Phi3ForSequenceClassification,
        Phi3ForTokenClassification,
        Phi3Model,
    )
    from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

    end_of_text_token = 32000

    class Phi3MiniWithStaticCache(torch.nn.Module):
        def __init__(self, model: Phi3ForCausalLM, batch_size: int, max_seq_len: int):
            super().__init__()
            self.model = model
            self.cache = StaticCache(config=model.config, max_cache_len=max_seq_len)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
        ) -> torch.FloatTensor:
            return self.model.forward(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
                past_key_values=self.cache,
            ).logits

        @staticmethod
        def generate(model: Phi3ForCausalLM, prompt_tokens: torch.LongTensor, max_seq_len: int) -> list[int]:
            model = Phi3MiniWithStaticCache(model, 1, max_seq_len + prompt_tokens.shape[-1])

            response_tokens = []

            for input_pos in range(prompt_tokens.shape[-1]):
                result = model.forward(
                    input_ids=prompt_tokens[:, input_pos : input_pos + 1],
                )
                response_tokens.append(prompt_tokens[0][input_pos].item())

            current_token = torch.argmax(result[:, -1, :], dim=-1).item()
            response_tokens.append(current_token)

            while current_token != end_of_text_token and len(response_tokens) < max_seq_len:
                result = model.forward(
                    input_ids=torch.tensor([[current_token]], dtype=torch.long),
                )
                current_token = torch.argmax(result[:, -1, :], dim=-1).item()
                response_tokens.append(current_token)

            return response_tokens


class Phi3ModelTester(CausalLMModelTester):
    config_class = Phi3Config
    if is_torch_available():
        base_model_class = Phi3Model
        causal_lm_class = Phi3ForCausalLM
        sequence_class = Phi3ForSequenceClassification
        token_class = Phi3ForTokenClassification


@require_torch
class Phi3ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (Phi3Model, Phi3ForCausalLM, Phi3ForSequenceClassification, Phi3ForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Phi3Model,
            "text-classification": Phi3ForSequenceClassification,
            "token-classification": Phi3ForTokenClassification,
            "text-generation": Phi3ForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    model_tester_class = Phi3ModelTester
    rotary_embedding_layer = Phi3RotaryEmbedding


@slow
@require_torch
class Phi3IntegrationTest(unittest.TestCase):
    def test_model_phi3_mini_4k_instruct_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor(
            [
                [8.9005, 8.5380, 12.0361, 9.1562, 7.4068, 10.2581, 7.8991, 7.2447,7.0626, 7.5760, 7.8315, 9.4076, 16.1104, 20.1290, 7.7500, 7.1947, 6.1550, 7.0563, 8.5344, 8.7248, 7.1359, 7.8237, 7.6817, 7.6395, 7.7924, 6.9702, 6.9097, 8.7074, 9.5768, 8.1145],
                [18.7090, 18.5701, 19.3660, 21.5171, 17.5042, 17.8716, 16.3554, 17.4617, 18.1623, 16.5641, 17.7547, 18.0193, 23.8355, 29.4481, 16.3864, 16.0560, 16.1543, 18.5507, 18.1343, 17.3883, 17.7422, 17.3012, 16.7657, 17.6874,17.9067, 16.8301, 16.2719, 18.3709, 19.0318, 16.7315],
            ]
        ).to(torch_device)  # fmt: skip

        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-4, atol=1e-4)

    def test_phi3_mini_4k_instruct_generation(self):
        model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        outputs = model.generate(inputs, max_new_tokens=32)
        output_text = tokenizer.batch_decode(outputs)

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> Certainly! Bananas and dragonfruits can be combined in various delicious ways. Here are some ideas for incorporating these fruits into your"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)

    def test_phi3_mini_4k_instruct_with_static_cache(self):
        model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        response_tokens = Phi3MiniWithStaticCache.generate(model, inputs, 64)

        output_text = tokenizer.batch_decode(torch.tensor([response_tokens], dtype=torch.long, device=torch_device))

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> Certainly! Bananas and dragonfruits can be combined in various delicious ways. Here are some"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)

    def test_model_phi3_mini_128k_instruct_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-128k-instruct").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor(
            [
                [10.6076, 10.6499, 14.0601, 19.8499, 15.1787, 19.3717, 19.9782, 17.0394, 15.7875, 18.1403, 19.2748, 12.6627, 20.2804, 24.5362, 18.8105, 15.3394, 12.1219, 15.9941, 19.0679, 16.4936, 17.0505, 16.8738, 17.3090, 16.6572, 16.8754, 16.6912, 15.1627, 18.8721, 19.6017, 18.5513],
                [16.2141, 18.7298, 17.4216, 21.9312, 17.7606, 17.6177, 16.7766, 17.9859, 18.4132, 17.4505, 18.6385, 18.5396, 23.6260, 28.7443, 16.1817, 15.5148, 16.0035, 18.6652, 18.3087, 17.2960, 17.8223, 17.7776, 16.8686, 17.4093, 17.8037, 17.2544, 16.7231, 18.6195, 19.6784, 16.6647],
            ]
        ).to(torch_device)  # fmt: skip

        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-4, atol=1e-4)

    def test_phi3_mini_128k_instruct_generation(self):
        model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-128k-instruct")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        outputs = model.generate(inputs, max_new_tokens=32)
        output_text = tokenizer.batch_decode(outputs)

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> Certainly! Bananas and dragonfruits can be combined in various delicious and nutritious ways. Here are some creative and healthy"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)

    def test_phi3_mini_128k_instruct_with_static_cache(self):
        model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-128k-instruct")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
            },
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        response_tokens = Phi3MiniWithStaticCache.generate(model, inputs, 64)

        output_text = tokenizer.batch_decode(torch.tensor([response_tokens], dtype=torch.long, device=torch_device))

        EXPECTED_OUTPUT = [
            "<|system|> You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.<|end|><|user|> Can you provide ways to eat combinations of bananas and dragonfruits?<|end|><|assistant|> Certainly! Bananas and dragonfruits can be combined in various delicious and nutritious ways"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)

    def test_phi3_mini_4k_sliding_window(self):
        """
        This tests that Phi3 doesn't deteriorate in quality for long context generations. Since Phi3 has
        sliding window attention, the test is tailored so that (context + max_new_tokens > sliding_window).
        See #33586 for more
        """
        model = Phi3ForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", device_map=torch_device, dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

        input_text = """
            <|user|>
            Tell me about Paris, France.<|end|>
            <|assistant|>
            Paris, the capital city of France, is renowned for its rich history, iconic landmarks, and vibrant culture. Known as "The City of Light," Paris is situated in the north-central part of the country along the Seine River.

            Here are some key aspects of Paris:

            1. Landmarks: Paris is home to numerous famous landmarks, including the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées. The Eiffel Tower, built in 1889, is an iconic symbol of Paris and attracts millions of tourists each year. The Louvre Museum, the world's largest art museum, houses thousands of works of art, including the Mona Lisa and the Venus de Milo.

            2. History: Paris has a rich history dating back to the 3rd century BC, when it was founded by a Celtic tribe called the Parisii. Over the centuries, the city has been influenced by various cultures, including the Romans, the Franks, and the Normans. The French Revolution in the late 18th century marked a significant turning point in Paris's history, leading to the establishment of the modern French Republic.

            3. Culture: Paris is a global center for art, fashion, gastronomy, and culture. The city is home to numerous museums, including the Centre Pompidou, Musée d'Orsay, and Musée Rodin. Paris is also known for its fashion industry, with many famous designers having their origins in the city. The city's cuisine is also highly regarded, with a focus on fresh ingredients, and a wide variety of dishes, including French classics like coq au vin, boeuf bourguignon, and crêpes.

            4. Architecture: Parisian architecture is characterized by its diverse styles, ranging from Gothic and Romanesque to Art Nouveau and Art Deco. The city's famous Haussmannian buildings, designed by Baron Haussmann in the mid-19th century, are known for their uniform facades, wrought-iron balconies, and large windows.

            5. Transportation: Paris has an extensive public transportation system, including the Paris Métro, RER (suburban trains), and buses. The city's iconic yellow taxis are also a popular mode of transportation.

            6. Language: The official language of Paris is French, and the city's residents are known for their charm and politeness.

            7. Festivals and Events: Paris hosts numerous festivals and events throughout the year, including the annual Bastille Day celebrations, the Paris Fashion Week, and the famous annual New Year's Eve fireworks on the Eiffel Tower.

            8. Geography: Paris is located in the north-central part of France, with the Seine River running through the city. The city's geography is characterized by rolling hills and picturesque parks, such as the Bois de Boulogne and the Jardin des Tuileries.

            9. Population: As of 2021, Paris has an estimated population of around 2.2 million residents, with the metropolitan area housing over 12 million people.

            In summary, Paris is a city steeped in history, culture, and art, with a unique blend of architectural styles and a vibrant atmosphere that continues to captivate millions of visitors each year.<|end|>
            <|user|>
            Please give me a list of 5 architectural landmarks in Paris, France.<|end|>
            <|assistant|>
            1. Eiffel Tower: Designed by Gustave Eiffel and completed in 1889, the Eiffel Tower is an iconic symbol of Paris and France. Standing at 324 meters tall, it was the tallest man-made structure in the world until the completion of the Chrysler Building in New York in 1930. The Eiffel Tower is made of wrought iron and offers visitors stunning views of the city from its three levels.

            2. Notre-Dame Cathedral: Located on the Île de la Cité, Notre-Dame Cathedral is a masterpiece of French Gothic architecture. Construction began in the 12th century and continued for over 200 years, with the cathedral's completion in the 14th century. The cathedral is famous for its intricate facade, stained-glass windows, and the iconic gargoyles and chimeras.

            3. Louvre Museum: Originally built as a fortress in the 12th century, the Louvre Museum is now the world's largest art museum and a historic monument in Paris. The museum's most famous landmark is the iconic glass pyramid entrance, designed by architect I. M. Pei in the 1980s. The Louvre houses over 380,000 works of art, including the Mona Lisa and the Venus de Milo.

            4. Sacré-Cœur Basilica: The Sacré-Cœur Basilica, also known as the Basilique du Sacré-Cœur, is a Roman Catholic church and minor basilica located at the summit of the butte Montmartre, the highest point in Paris. The basilica was designed by Paul Abadie and dedicated in 1914. Its white domes and lavender-colored travertine stone make it a distinctive landmark in the Paris skyline.

            5. Arc de Triomphe: The Arc de Triomphe is a monumental structure located at the western end of the Champs-Élysées. Commissioned by Napoleon in 1806, the Arc was designed by Jean-François-Thérèse Chalgrin and completed in 1836. The monument honors those who fought and died for France during the French Revolutionary and Napoleonic Wars. The Arc features sculptural reliefs and inscriptions, and its façade is adorned with the names of 357 generals and 660 soldiers.

            These five architectural landmarks showcase the diverse styles and historical periods of Paris, from Gothic to Neoclassical, and from the 19th to the 20th centuries. Each landmark has its unique features and contributes to the city's rich architectural heritage.<|end|>
            <|user|>
            Please give me a list of 10 famous items displayed in the Louvre Museum. Thanks!<|end|>
            <|assistant|>
            1. Mona Lisa: The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is arguably the most famous painting in the world. The portrait is known for its enigmatic smile and masterful use of sfumato, a technique that creates a soft, hazy effect.

            2. Venus de Milo: This ancient Greek statue, believed to have been created around 130-100 BC, is a masterpiece of Hellenistic sculpture. The Venus de Milo is renowned for its graceful beauty and the mystery surrounding its missing arms.

            3. Winged Victory of Samothrace: This Hellenistic sculpture, dating back to the 2nd century BC, depicts the Greek goddess Nike, the personification of victory. The sculpture is celebrated for its dynamic movement and intricate details.

            4. Liberty Leading the People: This iconic painting by Eugène Delacroix, created in 1830, commemorates the July Revolution in France. The artwork depicts a woman personifying Liberty leading a group of revolutionaries over the bodies of the fallen.

            5. The Wedding at Cana: A 1516 painting by Veronese, The Wedding at Cana is a large-scale work that depicts the biblical story of Jesus turning water into wine at a wedding feast. The painting is known for its vibrant colors and intricate details.

            6. The Raft of the Medusa: This 1819 painting by Théodore Géricault is a powerful depiction of the aftermath of the shipwreck of the French frigate Méduse. The painting is famous for its dramatic composition and emotional intensity.

            7. The Coronation of Napoleon: This 1805 painting by Jacques-Louis David portrays the coronation of Napoleon Bonaparte as Emperor of the French. The artwork is a masterpiece of neoclassical style and captures the pomp and ceremony of the event.

            8. The Death of Socrates: A 1914 painting by Pablo Picasso, The Death of Socrates depicts the moment when the ancient Greek philosopher Socrates drinks the poison hemlock, as ordered by the Athenian government. The painting is a powerful expression of the existential themes associated with Socrates' death.

            9. The Turkish Bath: A 1887 painting by Puvis de Chavannes, The Turkish Bath is a large-scale work that depicts a scene of relaxation and leisity in a traditional Turkish bath. The painting is known for its subdued color palette and serene atmosphere.

            10. The Wedding at Cana: A 1901 painting by Giovanni Boldini, The Wedding at Cana is a lively depiction of a festive Italian wedding. The painting is characterized by Boldini's signature flamboyant style and his ability to capture the essence of the subjects with a sense of movement and energy.

            These 10 famous items in the Louvre Museum showcase a diverse range of artistic styles and periods, from ancient Greek and Roman sculptures to Renaissance and 19th-century paintings. The collection at the Louvre is one of the world's largest and most important museums, housing over 800,000 works of art, and these famous items are just a small selection of the museum's vast collection.<|end|>
            <|user|>
            Please give me a list of 10 famous French dishes.<|end|>
            <|assistant|>
            """

        inputs = tokenizer(input_text, return_tensors="pt").to(device=torch_device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        output_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        EXPECTED_OUTPUT = [
            '1. Coq au Vin: Coq au Vin is a classic French dish that translates to "rooster in wine." The dish consists of chicken braised with wine, lardons, mushrooms, and garlic. It is a hearty and flavorful dish that is often served with potatoes or rice.\n\n            2. Boeuf Bourguignon: Boeuf Bourguignon is a traditional French beef stew that'
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)

    @slow
    def test_export_static_cache(self):
        from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_4

        if not is_torch_greater_or_equal_than_2_4:
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        model_id = "microsoft/Phi-4-mini-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id, pad_token="</s>", padding_side="right")

        expected_text_completions = Expectations(
            {
                ("rocm", (9, 5)): ["You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user. A 45-year-old patient with a 10-year history of type 2 diabetes mellitus presents with a 2-year history of progressive, non-healing, and painful, 2.5 cm"],
                ("cuda", None): ["You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user. A 45-year-old patient with a 10-year history of type 2 diabetes mellitus, who is currently on metformin and a SGLT2 inhibitor, presents with a 2-year history"],
            }
        )  # fmt: skip
        EXPECTED_TEXT_COMPLETION = expected_text_completions.get_expectation()
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load config
        config = AutoConfig.from_pretrained(model_id)
        # NOTE: To make the model exportable we need to set the rope scaling to default to avoid hitting
        # the data-dependent control flow in _longrope_frequency_update. Alternatively, we can rewrite
        # that function to avoid the data-dependent control flow.
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            config.rope_scaling["type"] = "default"

        # Load model
        device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompt = [
            "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."
        ]
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
        exported_program = exportable_module.export(
            input_ids=prompt_token_ids,
            cache_position=torch.arange(prompt_token_ids.shape[-1], dtype=torch.long, device=model.device),
        )
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
