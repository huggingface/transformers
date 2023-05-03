import requests


try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from transformers.tools.base import Tool


TEXT_DOWNLOAD_DESCRIPTION = (
    "This is a tool that downloads a file from a `url`. It takes the `url` as input, and returns the text"
    " contained in the file."
)


class TextDownloadTool(Tool):
    def __call__(self, url):
        if BeautifulSoup is None:
            raise ImportError("Please install bs4 in order to use this tool.")
        return BeautifulSoup(requests.get(url).text, features="html.parser").get_text()
