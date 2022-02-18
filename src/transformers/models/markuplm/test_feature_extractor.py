from transformers import MarkupLMFeatureExtractor


feature_extractor = MarkupLMFeatureExtractor()

html_string = """<HTML>

<HEAD>
  <TITLE>sample document</TITLE>
</HEAD>

<BODY BGCOLOR="FFFFFF">
  <HR>
  <a href="http://google.com">Goog</a>
  <H1>This is one header</H1>
  <H2>This is a another Header</H2>
  <P>Travel from
    <P>
      <B>SFO to JFK</B>
      <BR>
      <B><I>on May 2, 2015 at 2:00 pm. For details go to confirm.com </I></B>
      <HR>
      <div style="color:#0000FF">
        <h3>Traveler <b> name </b> is
        <p> John Doe </p>
      </div>"""

encoding = feature_extractor(html_string)
for k, v in encoding.items():
    print(k, v)
