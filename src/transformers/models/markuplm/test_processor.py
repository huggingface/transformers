from transformers import MarkupLMFeatureExtractor, MarkupLMProcessor, MarkupLMTokenizer


feature_extractor = MarkupLMFeatureExtractor()
tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base")

processor = MarkupLMProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def get_html_strings():
    html_string_1 = """<HTML>

    <HEAD> <TITLE>sample document</TITLE> </HEAD>

    <BODY BGCOLOR="FFFFFF"> <HR> <a href="http://google.com">Goog</a> <H1>This is one header</H1> <H2>This is a another
    Header</H2> <P>Travel from
        <P> <B>SFO to JFK</B> <BR> <B><I>on May 2, 2015 at 2:00 pm. For details go to confirm.com </I></B> <HR> <div
        style="color:#0000FF">
            <h3>Traveler <b> name </b> is <p> John Doe </p>
        </div>"""

    html_string_2 = """
    <!DOCTYPE html> <html> <body>

    <h1>My First Heading</h1> <p>My first paragraph.</p>

    </body> </html>
    """

    return [html_string_1, html_string_2]


html_strings = get_html_strings()

# not batched
question = "What's his name?"
input_processor = processor(html_strings[0], question, return_tensors="pt")

decoding = processor.decode(input_processor.input_ids.squeeze().tolist())
print("Actual decoding:", decoding)
