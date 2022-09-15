from transformers import MarkupLMFeatureExtractor, MarkupLMProcessor, MarkupLMTokenizer


feature_extractor = MarkupLMFeatureExtractor()
tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base")

processor = MarkupLMProcessor(feature_extractor, tokenizer)


def prepare_html_string():
    html_string = """
    <!DOCTYPE html> <html> <head> <title>Page Title</title> </head> <body>

    <h1>This is a Heading</h1> <p>This is a paragraph.</p>

    </body> </html>
    """

    return html_string


encoding = processor(prepare_html_string())
# for k, v in encoding.items():
#     print(k, v)

print(encoding.input_ids)

print(processor.decode(encoding.input_ids[0]))
