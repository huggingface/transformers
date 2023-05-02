import requests
from bs4 import BeautifulSoup

from transformers.tools.base import Tool


TEXT_DOWNLOAD_DESCRIPTION = (
    "This is a tool that downloads a file from a `url`. It takes the `url` as input, and returns the text"
    " contained in the file."
)


class TextDownload(Tool):
    def __call__(self, url):
        return BeautifulSoup(requests.get(url).text, features="html.parser").get_text()
