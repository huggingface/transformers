# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import gc
import unittest

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
    Conversation,
    ConversationalPipeline,
    TFAutoModelForCausalLM,
    pipeline,
)
from transformers.testing_utils import (
    backend_empty_cache,
    is_pipeline_test,
    is_torch_available,
    require_tf,
    require_torch,
    slow,
    torch_device,
)

from .test_pipelines_common import ANY


@is_pipeline_test
class ConversationalPipelineTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        if is_torch_available():
            backend_empty_cache(torch_device)

    model_mapping = dict(
        list(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.items())
        if MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
        else [] + list(MODEL_FOR_CAUSAL_LM_MAPPING.items())
        if MODEL_FOR_CAUSAL_LM_MAPPING
        else []
    )
    tf_model_mapping = dict(
        list(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.items())
        if TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
        else [] + list(TF_MODEL_FOR_CAUSAL_LM_MAPPING.items())
        if TF_MODEL_FOR_CAUSAL_LM_MAPPING
        else []
    )

    def get_test_pipeline(self, model, tokenizer, processor):
        conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)
        return conversation_agent, [Conversation("Hi there!")]

    def run_pipeline_test(self, conversation_agent, _):
        # Simple
        outputs = conversation_agent(Conversation("Hi there!"), max_new_tokens=5)
        self.assertEqual(
            outputs,
            Conversation([{"role": "user", "content": "Hi there!"}, {"role": "assistant", "content": ANY(str)}]),
        )

        # Single list
        outputs = conversation_agent([Conversation("Hi there!")], max_new_tokens=5)
        self.assertEqual(
            outputs,
            Conversation([{"role": "user", "content": "Hi there!"}, {"role": "assistant", "content": ANY(str)}]),
        )

        # Batch
        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")
        self.assertEqual(len(conversation_1), 1)
        self.assertEqual(len(conversation_2), 1)

        outputs = conversation_agent([conversation_1, conversation_2], max_new_tokens=5)
        self.assertEqual(outputs, [conversation_1, conversation_2])
        self.assertEqual(
            outputs,
            [
                Conversation(
                    [
                        {"role": "user", "content": "Going to the movies tonight - any suggestions?"},
                        {"role": "assistant", "content": ANY(str)},
                    ],
                ),
                Conversation(
                    [
                        {"role": "user", "content": "What's the last book you have read?"},
                        {"role": "assistant", "content": ANY(str)},
                    ]
                ),
            ],
        )

        # One conversation with history
        conversation_2.add_message({"role": "user", "content": "Why do you recommend it?"})
        outputs = conversation_agent(conversation_2, max_new_tokens=5)
        self.assertEqual(outputs, conversation_2)
        self.assertEqual(
            outputs,
            Conversation(
                [
                    {"role": "user", "content": "What's the last book you have read?"},
                    {"role": "assistant", "content": ANY(str)},
                    {"role": "user", "content": "Why do you recommend it?"},
                    {"role": "assistant", "content": ANY(str)},
                ]
            ),
        )

    @require_torch
    @slow
    def test_integration_torch_conversation(self):
        # When
        conversation_agent = pipeline(task="conversational", device=torch_device)
        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")
        # Then
        self.assertEqual(len(conversation_1.past_user_inputs), 0)
        self.assertEqual(len(conversation_2.past_user_inputs), 0)
        # When
        result = conversation_agent([conversation_1, conversation_2], do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, [conversation_1, conversation_2])
        self.assertEqual(len(result[0].past_user_inputs), 1)
        self.assertEqual(len(result[1].past_user_inputs), 1)
        self.assertEqual(len(result[0].generated_responses), 1)
        self.assertEqual(len(result[1].generated_responses), 1)
        self.assertEqual(result[0].past_user_inputs[0], "Going to the movies tonight - any suggestions?")
        self.assertEqual(result[0].generated_responses[0], "The Big Lebowski")
        self.assertEqual(result[1].past_user_inputs[0], "What's the last book you have read?")
        self.assertEqual(result[1].generated_responses[0], "The Last Question")
        # When
        conversation_2.add_user_input("Why do you recommend it?")
        result = conversation_agent(conversation_2, do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, conversation_2)
        self.assertEqual(len(result.past_user_inputs), 2)
        self.assertEqual(len(result.generated_responses), 2)
        self.assertEqual(result.past_user_inputs[1], "Why do you recommend it?")
        self.assertEqual(result.generated_responses[1], "It's a good book.")

    @require_torch
    @slow
    def test_integration_torch_conversation_truncated_history(self):
        # When
        conversation_agent = pipeline(task="conversational", min_length_for_response=24, device=torch_device)
        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        # Then
        self.assertEqual(len(conversation_1.past_user_inputs), 0)
        # When
        result = conversation_agent(conversation_1, do_sample=False, max_length=36)
        # Then
        self.assertEqual(result, conversation_1)
        self.assertEqual(len(result.past_user_inputs), 1)
        self.assertEqual(len(result.generated_responses), 1)
        self.assertEqual(result.past_user_inputs[0], "Going to the movies tonight - any suggestions?")
        self.assertEqual(result.generated_responses[0], "The Big Lebowski")
        # When
        conversation_1.add_user_input("Is it an action movie?")
        result = conversation_agent(conversation_1, do_sample=False, max_length=36)
        # Then
        self.assertEqual(result, conversation_1)
        self.assertEqual(len(result.past_user_inputs), 2)
        self.assertEqual(len(result.generated_responses), 2)
        self.assertEqual(result.past_user_inputs[1], "Is it an action movie?")
        self.assertEqual(result.generated_responses[1], "It's a comedy.")

    @require_torch
    def test_small_model_pt(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)
        conversation = Conversation("hello")
        output = conversation_agent(conversation)
        self.assertEqual(output, Conversation(past_user_inputs=["hello"], generated_responses=["Hi"]))

    @require_tf
    def test_small_model_tf(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = TFAutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)
        conversation = Conversation("hello")
        output = conversation_agent(conversation)
        self.assertEqual(output, Conversation(past_user_inputs=["hello"], generated_responses=["Hi"]))

    @require_torch
    @slow
    def test_integration_torch_conversation_dialogpt_input_ids(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)

        conversation_1 = Conversation("hello")
        inputs = conversation_agent.preprocess(conversation_1)
        self.assertEqual(inputs["input_ids"].tolist(), [[31373, 50256]])

        conversation_2 = Conversation("how are you ?", past_user_inputs=["hello"], generated_responses=["Hi there!"])
        inputs = conversation_agent.preprocess(conversation_2)
        self.assertEqual(
            inputs["input_ids"].tolist(), [[31373, 50256, 17250, 612, 0, 50256, 4919, 389, 345, 5633, 50256]]
        )

    @unittest.skip("Model is curently gated")
    @require_torch
    @slow
    def test_integration_torch_conversation_llama2_input_ids(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_default_system_prompt=True)

        conversation = Conversation(
            "What is so great about #1?",
            past_user_inputs=["I am going to Paris, what should I see?"],
            generated_responses=[
                """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."""
            ],
        )
        inputs = tokenizer._build_conversation_input_ids(conversation)
        EXPECTED_INPUTS_IDS = [ 1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 263, 8444, 29892, 3390, 1319, 322, 15993, 20255, 29889, 29849, 1234, 408, 1371, 3730, 408, 1950, 29892, 1550, 1641, 9109, 29889, 29871, 3575, 6089, 881, 451, 3160, 738, 10311, 1319, 29892, 443, 621, 936, 29892, 11021, 391, 29892, 7916, 391, 29892, 304, 27375, 29892, 18215, 29892, 470, 27302, 2793, 29889, 3529, 9801, 393, 596, 20890, 526, 5374, 635, 443, 5365, 1463, 322, 6374, 297, 5469, 29889, 13, 13, 3644, 263, 1139, 947, 451, 1207, 738, 4060, 29892, 470, 338, 451, 2114, 1474, 16165, 261, 296, 29892, 5649, 2020, 2012, 310, 22862, 1554, 451, 1959, 29889, 960, 366, 1016, 29915, 29873, 1073, 278, 1234, 304, 263, 1139, 29892, 3113, 1016, 29915, 29873, 6232, 2089, 2472, 29889, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 29902, 626, 2675, 304, 3681, 29892, 825, 881, 306, 1074, 29973, 518, 29914, 25580, 29962, 3681, 29892, 278, 7483, 310, 3444, 29892, 338, 2998, 363, 967, 380, 27389, 11258, 29892, 1616, 19133, 29879, 29892, 15839, 2982, 22848, 29892, 322, 6017, 7716, 25005, 29889, 2266, 526, 777, 310, 278, 2246, 19650, 1953, 304, 1074, 297, 3681, 29901, 13, 13, 29896, 29889, 450, 382, 2593, 295, 23615, 29901, 450, 9849, 293, 382, 2593, 295, 23615, 338, 697, 310, 278, 1556, 5936, 13902, 2982, 22848, 297, 278, 3186, 322, 16688, 2078, 271, 400, 5086, 8386, 310, 278, 4272, 29889, 13, 29906, 29889, 450, 4562, 12675, 6838, 29901, 450, 4562, 12675, 338, 697, 310, 278, 3186, 29915, 29879, 10150, 322, 1556, 13834, 19133, 29879, 29892, 27261, 385, 21210, 573, 4333, 310, 1616, 322, 24238, 29879, 29892, 3704, 278, 2598, 29874, 29420, 29889, 13, 29941, 29889, 24337, 29899, 29928, 420, 315, 21471, 29901, 910, 9560, 274, 21471, 338, 697, 310, 278, 1556, 13834, 2982, 22848, 297, 3681, 322, 338, 2998, 363, 967, 22883, 293, 11258, 322, 380, 27389, 380, 7114, 12917, 5417, 29889, 13, 13, 1349, 968, 526, 925, 263, 2846, 310, 278, 1784, 19650, 1953, 393, 3681, 756, 304, 5957, 29889, 2973, 577, 1568, 304, 1074, 322, 437, 29892, 372, 29915, 29879, 694, 4997, 393, 3681, 338, 697, 310, 278, 1556, 5972, 6282, 391, 15422, 800, 297, 278, 3186, 29889, 29871, 2, 1, 518, 25580, 29962, 1724, 338, 577, 2107, 1048, 396, 29896, 29973, 518, 29914, 25580, 29962]  # fmt: skip
        self.assertEqual(inputs, EXPECTED_INPUTS_IDS)

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)
        EXPECTED_TEXT = "what topic you want to focus on and create content around it. This will help you stand out from other creators and attract a specific audience.\n\nStep 2: Set Up Your Channel\nCreate your YouTube account and customize your channel with your branding and logo. Make sure your channel name and profile picture are consistent with your niche.\n\nStep 3: Plan Your Content\nDevelop a content strategy that includes the type of content you want to create, how often you will post, and when you will post. Consider creating a content calendar to help you stay organized.\n\nStep 4: Invest in Quality Equipment\nInvest in good quality camera and microphone equipment to ensure your videos look and sound professional. You don't need to break the bank, but investing in good equipment will make a big difference in the quality of your videos.\n\nStep 5: Optimize Your Videos for Search\nUse keywords in your video titles, descriptions, and tags to help people find your videos when they search for topics related to your niche"
        conversation = Conversation(
            "<<SYS>>\n Only answer with emojis, and charades\n<</SYS>>\n\nHow can I build a house in 10 steps?"
        )
        result = conversation_agent(conversation)
        self.assertEqual(result.generated_responses[-1], EXPECTED_TEXT)

    @require_torch
    @slow
    def test_integration_torch_conversation_blenderbot_400M_input_ids(self):
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)

        # test1
        conversation_1 = Conversation("hello")
        inputs = conversation_agent.preprocess(conversation_1)
        self.assertEqual(inputs["input_ids"].tolist(), [[1710, 86, 2]])

        # test2
        conversation_1 = Conversation(
            "I like lasagne.",
            past_user_inputs=["hello"],
            generated_responses=[
                " Do you like lasagne? It is a traditional Italian dish consisting of a shepherd's pie."
            ],
        )
        inputs = conversation_agent.preprocess(conversation_1)
        self.assertEqual(
            inputs["input_ids"].tolist(),
            [
                # This should be compared with the same conversation on ParlAI `safe_interactive` demo.
                [
                    1710,  # hello
                    86,
                    228,  # Double space
                    228,
                    946,
                    304,
                    398,
                    6881,
                    558,
                    964,
                    38,
                    452,
                    315,
                    265,
                    6252,
                    452,
                    322,
                    968,
                    6884,
                    3146,
                    278,
                    306,
                    265,
                    617,
                    87,
                    388,
                    75,
                    341,
                    286,
                    521,
                    21,
                    228,  # Double space
                    228,
                    281,  # I like lasagne.
                    398,
                    6881,
                    558,
                    964,
                    21,
                    2,  # EOS
                ],
            ],
        )

    @require_torch
    @slow
    def test_integration_torch_conversation_blenderbot_400M(self):
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer)

        conversation_1 = Conversation("hello")
        result = conversation_agent(
            conversation_1,
        )
        self.assertEqual(
            result.generated_responses[0],
            # ParlAI implementation output, we have a different one, but it's our
            # second best, you can check by using num_return_sequences=10
            # " Hello! How are you? I'm just getting ready to go to work, how about you?",
            " Hello! How are you doing today? I just got back from a walk with my dog.",
        )

        conversation_1 = Conversation("Lasagne   hello")
        result = conversation_agent(conversation_1, encoder_no_repeat_ngram_size=3)
        self.assertEqual(
            result.generated_responses[0],
            " Do you like lasagne? It is a traditional Italian dish consisting of a shepherd's pie.",
        )

        conversation_1 = Conversation(
            "Lasagne   hello   Lasagne is my favorite Italian dish. Do you like lasagne?   I like lasagne."
        )
        result = conversation_agent(
            conversation_1,
            encoder_no_repeat_ngram_size=3,
        )
        self.assertEqual(
            result.generated_responses[0],
            " Me too. I like how it can be topped with vegetables, meats, and condiments.",
        )

    @require_torch
    @slow
    def test_integration_torch_conversation_encoder_decoder(self):
        # When
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot_small-90M")
        conversation_agent = ConversationalPipeline(model=model, tokenizer=tokenizer, device=torch_device)

        conversation_1 = Conversation("My name is Sarah and I live in London")
        conversation_2 = Conversation("Going to the movies tonight, What movie would you recommend? ")
        # Then
        self.assertEqual(len(conversation_1.past_user_inputs), 0)
        self.assertEqual(len(conversation_2.past_user_inputs), 0)
        # When
        result = conversation_agent([conversation_1, conversation_2], do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, [conversation_1, conversation_2])
        self.assertEqual(len(result[0].past_user_inputs), 1)
        self.assertEqual(len(result[1].past_user_inputs), 1)
        self.assertEqual(len(result[0].generated_responses), 1)
        self.assertEqual(len(result[1].generated_responses), 1)
        self.assertEqual(result[0].past_user_inputs[0], "My name is Sarah and I live in London")
        self.assertEqual(
            result[0].generated_responses[0],
            "hi sarah, i live in london as well. do you have any plans for the weekend?",
        )
        self.assertEqual(
            result[1].past_user_inputs[0], "Going to the movies tonight, What movie would you recommend? "
        )
        self.assertEqual(
            result[1].generated_responses[0], "i don't know... i'm not really sure. what movie are you going to see?"
        )
        # When
        conversation_1.add_user_input("Not yet, what about you?")
        conversation_2.add_user_input("What's your name?")
        result = conversation_agent([conversation_1, conversation_2], do_sample=False, max_length=1000)
        # Then
        self.assertEqual(result, [conversation_1, conversation_2])
        self.assertEqual(len(result[0].past_user_inputs), 2)
        self.assertEqual(len(result[1].past_user_inputs), 2)
        self.assertEqual(len(result[0].generated_responses), 2)
        self.assertEqual(len(result[1].generated_responses), 2)
        self.assertEqual(result[0].past_user_inputs[1], "Not yet, what about you?")
        self.assertEqual(result[0].generated_responses[1], "i don't have any plans yet. i'm not sure what to do yet.")
        self.assertEqual(result[1].past_user_inputs[1], "What's your name?")
        self.assertEqual(result[1].generated_responses[1], "i don't have a name, but i'm going to see a horror movie.")

    @require_torch
    @slow
    def test_from_pipeline_conversation(self):
        model_id = "facebook/blenderbot_small-90M"

        # from model id
        conversation_agent_from_model_id = pipeline("conversational", model=model_id, tokenizer=model_id)

        # from model object
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_id)
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_id)
        conversation_agent_from_model = pipeline("conversational", model=model, tokenizer=tokenizer)

        conversation = Conversation("My name is Sarah and I live in London")
        conversation_copy = Conversation("My name is Sarah and I live in London")

        result_model_id = conversation_agent_from_model_id([conversation])
        result_model = conversation_agent_from_model([conversation_copy])

        # check for equality
        self.assertEqual(
            result_model_id.generated_responses[0],
            "hi sarah, i live in london as well. do you have any plans for the weekend?",
        )
        self.assertEqual(
            result_model_id.generated_responses[0],
            result_model.generated_responses[0],
        )
