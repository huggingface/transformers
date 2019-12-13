""" Link tester.

This little utility reads all the python files in the repository,
scans for links pointing to S3 and tests the links one by one. Raises an error
at the end of the scan if at least one link was reported broken.
"""
import os
import re
import sys

import requests


REGEXP_FIND_S3_LINKS = r"""([\"'])(https:\/\/s3)(.*)?\1"""


def list_python_files_in_repository():
    """ List all python files in the repository.

    This function assumes that the script is executed in the root folder.
    """
    source_code_files = []
    for path, subdirs, files in os.walk("."):
        if "templates" in path:
            continue
        for name in files:
            if ".py" in name and ".pyc" not in name:
                path_to_files = os.path.join(path, name)
                source_code_files.append(path_to_files)

    return source_code_files


def find_all_links(file_paths):
    links = []
    for path in file_paths:
        links += scan_code_for_links(path)

    return links


def scan_code_for_links(source):
    """ Scans the file to find links using a regular expression.
    Returns a list of links.
    """
    with open(source, 'r') as content:
        content = content.read()
        raw_links = re.findall(REGEXP_FIND_S3_LINKS, content)
        links = [prefix + suffix for _, prefix, suffix in raw_links]

    return links


def check_all_links(links):
    """ Check that the provided links are valid.

    Links are considered valid if a HEAD request to the server
    returns a 200 status code.
    """
    broken_links = []
    for link in links:
        head = requests.head(link)
        if head.status_code != 200:
            broken_links.append(link)

    return broken_links


if __name__ == "__main__":
    file_paths = list_python_files_in_repository()
    links = find_all_links(file_paths)
    broken_links = check_all_links(links)
    print("Looking for broken links to pre-trained models/configs/tokenizers...")
    if broken_links:
        print("The following links did not respond:")
        for link in broken_links:
            print("- {}".format(link))
        sys.exit(1)
    print("All links are ok.")
