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

# CASE1: non-batched
inputs = processor(html_strings[0], return_tensors="pt")
decoding = processor.decode(inputs.input_ids.squeeze().tolist())

# CASE 1: batched
inputs = processor(html_strings, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
decoding = processor.decode(inputs.input_ids[1].tolist())

# CASE 2: non-batched
processor.parse_html = False
nodes = (["hello", "world", "how", "are"],)
xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
inputs = processor(nodes=nodes, xpaths=xpaths, return_tensors="pt")

# CASE 2: batched
nodes = [["hello", "world"], ["my", "name", "is"]]
xpaths = [
    ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
    ["html/body", "html/body/div", "html/body"],
]

inputs = processor(nodes=nodes, xpaths=xpaths, padding=True, return_tensors="pt")

print(processor.decode(inputs.input_ids[0].tolist()))

# CASE 3: not batched
nodes = ["hello", "world", "how", "are"]
xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
node_labels = [0, 1, 2, 3]
inputs = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")

# CASE 3: batched
nodes = [["hello", "world"], ["my", "name", "is"]]
xpaths = [
    ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
    ["html/body", "html/body/div", "html/body"],
]
node_labels = [[0, 1], [67, 2]]
inputs = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")

# for id, label in zip(inputs.input_ids[1], inputs.labels[1]):
#     print(processor.decode([id.item()]), label.item())

# CASE 4: not batched
processor.parse_html = True
inputs = processor(html_strings[0], questions="how are you?", return_tensors="pt")

# decoding = processor.decode(inputs.input_ids.squeeze().tolist())
# print("Actual decoding:", decoding)

# CASE 4: batched
inputs = processor(
    html_strings,
    questions=["how are you?", "what's your name?"],
    padding="max_length",
    max_length=20,
    truncation=True,
    return_tensors="pt",
)

# decoding = processor.decode(inputs.input_ids[1].tolist())
# print("Actual decoding:", decoding)

# CASE 5: not batched
processor.parse_html = False
nodes = ["hello", "world", "how", "are"]
xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"]
inputs = processor(nodes=nodes, xpaths=xpaths, questions="how are you?", return_tensors="pt")

decoding = processor.decode(inputs.input_ids[0].tolist())
print("Actual decoding:", decoding)

# CASE 5: batched
nodes = [["hello", "world"], ["my", "name", "is"]]
xpaths = [
    ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
    ["html/body", "html/body/div", "html/body"],
]

inputs = processor(
    nodes=nodes,
    xpaths=xpaths,
    questions=["how are you?", "what's your name?"],
    padding="max_length",
    max_length=20,
    truncation=True,
    return_tensors="pt",
)
