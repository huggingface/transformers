import unittest

import pytest

from transformers import pipeline
from transformers.pipelines import PipelineWarning
from transformers.testing_utils import require_torch, slow

from .test_pipelines_common import MonoInputPipelineCommonMixin


class SentimentAnalysisPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "sentiment-analysis"
    small_models = [
        "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
    ]  # Default model - Models tested without the @slow decorator
    large_models = [None]  # Models tested with the @slow decorator
    mandatory_keys = {"label", "score"}  # Keys which should be in the output

    @require_torch
    def test_input_too_long(self):
        model = self.small_models[0]
        pipe = pipeline(self.pipeline_task, model=model)

        with pytest.warns(PipelineWarning, match=r".*truncated.*"):
            pipe(
                """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. At ultrices mi tempus imperdiet nulla malesuada. Nam libero justo laoreet sit amet cursus. Libero volutpat sed cras ornare arcu dui. Nunc aliquet bibendum enim facilisis gravida neque convallis. Odio pellentesque diam volutpat commodo sed egestas. Malesuada nunc vel risus commodo viverra maecenas accumsan. Id semper risus in hendrerit gravida. Habitant morbi tristique senectus et netus. Habitant morbi tristique senectus et netus. Id semper risus in hendrerit gravida rutrum quisque non. Faucibus vitae aliquet nec ullamcorper sit amet risus nullam eget. Non pulvinar neque laoreet suspendisse interdum consectetur libero id. Quis commodo odio aenean sed adipiscing diam. Ut diam quam nulla porttitor massa id. Posuere lorem ipsum dolor sit amet consectetur. Sollicitudin nibh sit amet commodo nulla facilisi nullam vehicula. Mattis rhoncus urna neque viverra justo nec. Odio ut sem nulla pharetra diam sit amet.

Magna fringilla urna porttitor rhoncus dolor. Lorem ipsum dolor sit amet consectetur adipiscing elit duis tristique. Sit amet consectetur adipiscing elit. Auctor elit sed vulputate mi sit. Ac turpis egestas sed tempus. Ut aliquam purus sit amet. Id semper risus in hendrerit gravida rutrum. A diam sollicitudin tempor id eu. Lorem ipsum dolor sit amet. Enim neque volutpat ac tincidunt. Dictum sit amet justo donec enim diam. Sapien faucibus et molestie ac. Dictum sit amet justo donec enim diam vulputate ut pharetra. Porttitor eget dolor morbi non arcu risus. Viverra nam libero justo laoreet. Consectetur purus ut faucibus pulvinar. Nunc mi ipsum faucibus vitae aliquet nec ullamcorper sit amet. Auctor eu augue ut lectus arcu. Ultricies mi eget mauris pharetra et ultrices. Volutpat diam ut venenatis tellus.

Tortor at auctor urna nunc id cursus. Massa eget egestas purus viverra accumsan in nisl. Sed lectus vestibulum mattis ullamcorper velit sed ullamcorper. Morbi tincidunt ornare massa eget egestas. Tincidunt vitae semper quis lectus nulla at. Viverra nam libero justo laoreet sit amet cursus sit. A iaculis at erat pellentesque. A pellentesque sit amet porttitor eget dolor morbi non. Massa sed elementum tempus egestas sed sed risus pretium quam. Ac turpis egestas maecenas pharetra convallis. Nisi quis eleifend quam adipiscing vitae proin.

Morbi enim nunc faucibus a. Vel quam elementum pulvinar etiam non quam. Egestas dui id ornare arcu odio ut. Ut ornare lectus sit amet est placerat in. Ut pharetra sit amet aliquam id diam. Arcu ac tortor dignissim convallis aenean et tortor at risus. Phasellus faucibus scelerisque eleifend donec pretium vulputate sapien nec sagittis. Vestibulum rhoncus est pellentesque elit ullamcorper dignissim. Eros in cursus turpis massa tincidunt. Non sodales neque sodales ut etiam sit. Ultricies lacus sed turpis tincidunt id aliquet. In nisl nisi scelerisque eu ultrices vitae auctor. Eget mi proin sed libero enim sed faucibus. Orci dapibus ultrices in iaculis nunc sed augue lacus. Commodo elit at imperdiet dui accumsan sit amet. Ac odio tempor orci dapibus. Ullamcorper morbi tincidunt ornare massa eget. Sed euismod nisi porta lorem mollis aliquam.

Dictumst vestibulum rhoncus est pellentesque elit ullamcorper dignissim cras tincidunt. Viverra suspendisse potenti nullam ac tortor vitae purus. Ornare massa eget egestas purus. Parturient montes nascetur ridiculus mus mauris vitae ultricies. Maecenas accumsan lacus vel facilisis. Consectetur lorem donec massa sapien faucibus et molestie. Elit ut aliquam purus sit amet luctus. Quis auctor elit sed vulputate mi sit amet mauris. Id leo in vitae turpis massa sed. Consequat ac felis donec et odio pellentesque diam volutpat commodo. Platea dictumst vestibulum rhoncus est pellentesque elit ullamcorper dignissim. Nec ultrices dui sapien eget mi proin sed libero. Quisque non tellus orci ac auctor augue.
"""
            )

    @require_torch
    @slow
    def test_input_too_long_roberta_openai_detector(self):
        model = "roberta-base-openai-detector"
        pipe = pipeline(self.pipeline_task, model=model)

        with pytest.warns(PipelineWarning, match=r".*truncated.*"):
            pipe(
                """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. At ultrices mi tempus imperdiet nulla malesuada. Nam libero justo laoreet sit amet cursus. Libero volutpat sed cras ornare arcu dui. Nunc aliquet bibendum enim facilisis gravida neque convallis. Odio pellentesque diam volutpat commodo sed egestas. Malesuada nunc vel risus commodo viverra maecenas accumsan. Id semper risus in hendrerit gravida. Habitant morbi tristique senectus et netus. Habitant morbi tristique senectus et netus. Id semper risus in hendrerit gravida rutrum quisque non. Faucibus vitae aliquet nec ullamcorper sit amet risus nullam eget. Non pulvinar neque laoreet suspendisse interdum consectetur libero id. Quis commodo odio aenean sed adipiscing diam. Ut diam quam nulla porttitor massa id. Posuere lorem ipsum dolor sit amet consectetur. Sollicitudin nibh sit amet commodo nulla facilisi nullam vehicula. Mattis rhoncus urna neque viverra justo nec. Odio ut sem nulla pharetra diam sit amet.

Magna fringilla urna porttitor rhoncus dolor. Lorem ipsum dolor sit amet consectetur adipiscing elit duis tristique. Sit amet consectetur adipiscing elit. Auctor elit sed vulputate mi sit. Ac turpis egestas sed tempus. Ut aliquam purus sit amet. Id semper risus in hendrerit gravida rutrum. A diam sollicitudin tempor id eu. Lorem ipsum dolor sit amet. Enim neque volutpat ac tincidunt. Dictum sit amet justo donec enim diam. Sapien faucibus et molestie ac. Dictum sit amet justo donec enim diam vulputate ut pharetra. Porttitor eget dolor morbi non arcu risus. Viverra nam libero justo laoreet. Consectetur purus ut faucibus pulvinar. Nunc mi ipsum faucibus vitae aliquet nec ullamcorper sit amet. Auctor eu augue ut lectus arcu. Ultricies mi eget mauris pharetra et ultrices. Volutpat diam ut venenatis tellus.

Tortor at auctor urna nunc id cursus. Massa eget egestas purus viverra accumsan in nisl. Sed lectus vestibulum mattis ullamcorper velit sed ullamcorper. Morbi tincidunt ornare massa eget egestas. Tincidunt vitae semper quis lectus nulla at. Viverra nam libero justo laoreet sit amet cursus sit. A iaculis at erat pellentesque. A pellentesque sit amet porttitor eget dolor morbi non. Massa sed elementum tempus egestas sed sed risus pretium quam. Ac turpis egestas maecenas pharetra convallis. Nisi quis eleifend quam adipiscing vitae proin.

Morbi enim nunc faucibus a. Vel quam elementum pulvinar etiam non quam. Egestas dui id ornare arcu odio ut. Ut ornare lectus sit amet est placerat in. Ut pharetra sit amet aliquam id diam. Arcu ac tortor dignissim convallis aenean et tortor at risus. Phasellus faucibus scelerisque eleifend donec pretium vulputate sapien nec sagittis. Vestibulum rhoncus est pellentesque elit ullamcorper dignissim. Eros in cursus turpis massa tincidunt. Non sodales neque sodales ut etiam sit. Ultricies lacus sed turpis tincidunt id aliquet. In nisl nisi scelerisque eu ultrices vitae auctor. Eget mi proin sed libero enim sed faucibus. Orci dapibus ultrices in iaculis nunc sed augue lacus. Commodo elit at imperdiet dui accumsan sit amet. Ac odio tempor orci dapibus. Ullamcorper morbi tincidunt ornare massa eget. Sed euismod nisi porta lorem mollis aliquam.

Dictumst vestibulum rhoncus est pellentesque elit ullamcorper dignissim cras tincidunt. Viverra suspendisse potenti nullam ac tortor vitae purus. Ornare massa eget egestas purus. Parturient montes nascetur ridiculus mus mauris vitae ultricies. Maecenas accumsan lacus vel facilisis. Consectetur lorem donec massa sapien faucibus et molestie. Elit ut aliquam purus sit amet luctus. Quis auctor elit sed vulputate mi sit amet mauris. Id leo in vitae turpis massa sed. Consequat ac felis donec et odio pellentesque diam volutpat commodo. Platea dictumst vestibulum rhoncus est pellentesque elit ullamcorper dignissim. Nec ultrices dui sapien eget mi proin sed libero. Quisque non tellus orci ac auctor augue.
"""
            )
