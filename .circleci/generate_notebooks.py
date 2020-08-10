import nbformat
import os
import re
import shutil

# Launch from root of repo
PATH_TO_DOCS = 'docs/source'
PATH_TO_DEST = 'docs'
DOC_BASE_URL = "https://huggingface.co/transformers/"

# These are the doc files converted, add any new tutorial to this list if you want it handled by the conversion
# script.
TUTORIAL_FILES = [
    "benchmarks.rst",
    "multilingual.rst",
    "perplexity.rst",
    "preprocessing.rst",
    "quicktour.rst",
    "task_summary.rst",
    "tokenizer_summary.rst",
    "training.rst"
]

###################################
# Parsing the rst file            #
###################################

# Re pattern that catches markdown titles.
_re_title = re.compile(r"^#+\s+(\S+)")
# Re pattern that catches rst blocks of the form `.. block_name::`.
_re_block = re.compile(r"^\.\.\s+(\S+)::")
# Re pattern that catches what's after the :: in rst blocks of the form `.. block_name:: something`.
_re_block_lang = re.compile(r"^\.\.\s+\S+::\s*(\S+)(\s+|$)")
# Re pattern that catchers section names like `.. _name:`.
_re_anchor_section = re.compile(r"^\.\.\s+_(\S+):")
# Re pattern that catches indentation at the start of a line.
_re_indent = re.compile(r"^(\s*)\S")


def split_blocks(lines):
    """ Read the lines of a doc file and group them by blocks."""
    blocks = []
    block_type = None
    current_block = []
    i = 0

    def _move_to_next_non_empty_line(i):
        while i < len(lines) and len(lines[i]) == 0:
            i += 1
        return i

    def _build_block(blocks, current_block, block_type):
        if len(current_block) > 0:
            while len(current_block[-1]) == 0:
                current_block = current_block[:-1]
            blocks.append(('\n'.join(current_block), block_type))
        return blocks, []

    while i < len(lines):
        line = lines[i]
        if _re_title.search(line) is not None:
            blocks, current_block = _build_block(blocks, current_block, "prose")
            blocks.append((line, "title"))
            i += 1
            i = _move_to_next_non_empty_line(i)
        elif _re_block.search(line) is not None:
            blocks, current_block = _build_block(blocks, current_block, "prose")
            block_type = _re_block.search(line).groups()[0]
            if _re_block_lang.search(line):
                block_type += " " + _re_block_lang.search(line).groups()[0]
            i += 1
            i = _move_to_next_non_empty_line(i)
            indent = _re_indent.search(lines[i]).groups()[0]
            if len(indent) > 0:
                while i < len(lines) and (lines[i].startswith(indent) or len(lines[i]) == 0):
                    current_block.append(lines[i])
                    i += 1
            blocks, current_block = _build_block(blocks, current_block, block_type)
        elif _re_anchor_section.search(line):
            blocks, current_block = _build_block(blocks, current_block, "prose")
            blocks.append((line, "anchor"))
            i += 1
            i = _move_to_next_non_empty_line(i)
        else:
            current_block.append(line)
            i += 1
    blocks, current_block = _build_block(blocks, current_block, "prose")
    return blocks


###################################
# Text formatting and cleaning    #
###################################

def process_titles(lines):
    """ Converts rst titles to markdown titles."""
    title_chars = """= - ` : ' " ~ ^ _ * + # < >""".split(" ")
    title_levels = {}
    new_lines = []
    for line in lines:
        if len(new_lines) > 0 and len(line) >= len(new_lines[-1]) and len(set(line)) == 1 and line[
            0] in title_chars and line != "::":
            char = line[0]
            level = title_levels.get(char, len(title_levels) + 1)
            if level not in title_levels:
                title_levels[char] = level
            new_lines[-1] = f"{'#' * level} {new_lines[-1]}"
        else:
            new_lines.append(line)
    return new_lines


# Re pattern to catch things inside ` ` in :obj:`thing`.
_re_obj = re.compile(r":obj:`([^`]+)`")
# Re pattern to catch things inside ` ` in :math:`thing`.
_re_math = re.compile(r":math:`([^`]+)`")
# Re pattern to catch things between single backquotes.
_re_single_backquotes = re.compile(r"(^|[^`])`([^`]+)`([^`]|$)")
# Re pattern to catch things between stars.
_re_stars = re.compile(r"\*([^\*]+)\*")
# Re pattern to catch things between double backquotes.
_re_double_backquotes = re.compile(r"``([^`]+)``")
# Re pattern to catch things inside ` ` in :func/class/meth:`thing`.
_re_func_class = re.compile(r":(?:func|class|meth):`([^`]+)`")


def convert_rst_formatting(text):
    """ Convert rst syntax for formatting to markdown in text."""

    # Remove :class:, :func: and :meth: markers. Simplify what's inside and put double backquotes
    # (to not be caught by the italic conversion).
    def _rep_func_class(match):
        name = match.groups()[0]
        splits = name.split('.')
        i = 0
        while i < len(splits) - 1 and not splits[i][0].isupper():
            i += 1
        return f"``{'.'.join(splits[i:])}``"

    text = _re_func_class.sub(_rep_func_class, text)
    # Remove :obj: markers. What's after is in a single backquotes so we put in double backquotes
    # (to not be caught by the italic conversion).
    text = _re_obj.sub(r"``\1``", text)
    # Remove :math: markers.
    text = _re_math.sub(r"$\1$", text)
    # Convert content in stars to bold
    text = _re_stars.sub(r'**\1**', text)
    # Convert content in single backquotes to italic.
    text = _re_single_backquotes.sub(r'\1*\2*\3', text)
    # Convert content in double backquotes to single backquotes.
    text = _re_double_backquotes.sub(r'`\1`', text)
    # Remove remaining ::
    text = re.sub(r"::\n", "", text)
    return text


# Re pattern to catch description and url in links of the form `description <url>`_.
_re_links = re.compile(r"`([^`]+\S)\s+</*([^/][^>`]*)>`_+")
# Re pattern to catch reference in links of the form :doc:`reference`.
_re_simple_doc = re.compile(r":doc:`([^`<]*)`")
# Re pattern to catch description and reference in links of the form :doc:`description <reference>`.
_re_doc_with_description = re.compile(r":doc:`([^`<]+\S)\s+</*([^/][^>`]*)>`")
# Re pattern to catch reference in links of the form :ref:`reference`.
_re_simple_ref = re.compile(r":ref:`([^`<]*)`")
# Re pattern to catch description and reference in links of the form :ref:`description <reference>`.
_re_ref_with_description = re.compile(r":ref:`([^`<]+\S)\s+<([^>]*)>`")


def convert_rst_links(text):
    """ Convert the rst links in text to markdown."""
    # Links of the form :doc:`page`
    text = _re_simple_doc.sub(r'[\1](' + DOC_BASE_URL + r'\1.html)', text)
    # Links of the form :doc:`text <page>`
    text = _re_doc_with_description.sub(r'[\1](' + DOC_BASE_URL + r'\2.html)', text)
    # Refs of the form :ref:`page`
    text = _re_simple_ref.sub(r'[\1](#\1)', text)
    # Refs of the form :ref:`text <page>`
    text = _re_ref_with_description.sub(r'[\1](#\2)', text)

    # Other links
    def _rep_links(match):
        text, url = match.groups()
        if not url.startswith('http'):
            url = DOC_BASE_URL + url
        return f"[{text}]({url})"

    text = _re_links.sub(_rep_links, text)
    return text


###################################
# Notes, math and reference       #
###################################

def remove_indentation(text):
    """ Remove the indendation found in the first line in text."""
    lines = text.split("\n")
    indent = _re_indent.search(lines[0]).groups()[0]
    new_lines = [line[len(indent):] for line in lines]
    return "\n".join(new_lines)


# For now we just do **NOTE_TYPE:** text, maybe there is some clever html solution to have something nicer.
def convert_to_note(text, note_type):
    """ Convert text to a note of note_type."""
    text = remove_indentation(text)
    lines = text.split("\n")
    new_lines = [f"> **{note_type.upper()}:** {lines[0]}"]
    new_lines += [f"> {line}" for line in lines[1:]]
    return "\n".join(new_lines)


def convert_math(text):
    """ Convert text to disaply mode LaTeX."""
    text = remove_indentation(text)
    return f"$${text}$$"


def convert_anchor(text):
    """ Convert text to an anchor that can be used in the notebook."""
    anchor_name = _re_anchor_section.search(text).groups()[0]
    return f"<a id='{anchor_name}'></a>"


###################################
# Images                          #
###################################

_re_attr_rst = re.compile(r"^\s*:(\S+):\s*(\S.*)$")


def convert_image(image_name, text, pref=None, origin_folder=None, dest_folder=None):
    """ Convert text to proper html code for image_name.
    Optionally copy image from origin_folder to dest_folder."""
    # Copy the image if necessary
    if origin_folder is not None and dest_folder is not None:
        origin_file = os.path.join(origin_folder, image_name)
        dest_file = os.path.join(dest_folder, image_name)
        if not os.path.isfile(dest_file):
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            shutil.copy(origin_file, dest_file)
    attrs = {'src': image_name if pref is None else os.path.join(pref, image_name)}
    for line in text.split("\n"):
        if _re_attr_rst.search(line) is not None:
            key, attr = _re_attr_rst.search(line).groups()
            attrs[key] = attr
    html = " ".join([f'{key}="{value}"' for key, value in attrs.items()])
    return f"<img {html}/>"


###################################
# Tables                          #
###################################

# Matches lines with a pattern of a table new line in rst.
_re_ignore_line_table = re.compile("^(\+[\-\s]+)+\+\s*$")
# Matches lines with a pattern of a table new line in rst, with a first column empty.
_re_ignore_line_table1 = re.compile("^\|\s+(\+[\-\s]+)+\+\s*$")
# Matches lines with a pattern of a first table line in rst.
_re_sep_line_table = re.compile("^(\+[=\s]+)+\+\s*$")


def convert_table(text):
    """ Convert a table in text from rst to markdown."""
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        if _re_ignore_line_table.search(line) is not None:
            continue
        if _re_ignore_line_table1.search(line) is not None:
            continue
        if _re_sep_line_table.search(line) is not None:
            line = line.replace('=', '-').replace('+', '|')
        new_lines.append(line)
    return "\n".join(new_lines)


###################################
# Code cleaning                   #
###################################

# Matches the pytorch code tag.
_re_pytorch = re.compile(r"## PYTORCH CODE")
# Matches the tensorflow code tag.
_re_tensorflow = re.compile(r"## TENSORFLOW CODE")


def split_frameworks(code):
    """ Split code between the two frameworks (if it has two versions) with PyTorch first."""
    if _re_pytorch.search(code) is None or _re_tensorflow.search(code) is None:
        return (code,)
    lines = code.split("\n")
    is_pytorch_first = _re_pytorch.search(lines[0]) is not None
    re_split = _re_tensorflow if is_pytorch_first else _re_pytorch
    i = 1
    while re_split.search(lines[i]) is None:
        i += 1
    j = i - 1
    while len(lines[j]) == 0:
        j -= 1
    return ("\n".join(lines[:j + 1]), "\n".join(lines[i:])) if is_pytorch_first else (
    "\n".join(lines[i:]), "\n".join(lines[:j + 1]))


# Matches any doctest pattern.
_re_doctest = re.compile(r"^(>>>|\.\.\.)")


def parse_code_and_output(code):
    """ Parse code to remove indentation, doctest prompts and split between source and theoretical output."""
    lines = code.split("\n")
    indent = _re_indent.search(lines[0]).groups()[0]
    has_doctest = False
    input_lines = []
    output_lines = []
    for line in lines:
        if len(line) > 0:
            line = line[len(indent):]
        if _re_doctest.search(line):
            has_doctest = True
            line = line[4:]
            input_lines.append(line)
        elif has_doctest:
            if len(line) > 0:
                output_lines.append(line)
        else:
            input_lines.append(line)
    return "\n".join(input_lines), "\n".join(output_lines)


###################################
# All together!                   #
###################################

def markdown_cell(md):
    """ Create a markdown cell with md inside."""
    return nbformat.notebooknode.NotebookNode({'cell_type': 'markdown', 'source': md, 'metadata': {}})


def code_cell(code, output=None):
    """ Create a code cell with `code` and optionally, `output`."""
    if output is None or len(output) == 0:
        outputs = []
    else:
        outputs = [nbformat.notebooknode.NotebookNode({
            'data': {'text/plain': output},
            'execution_count': None,
            'metadata': {},
            'output_type': 'execute_result'
        })]
    return nbformat.notebooknode.NotebookNode(
        {'cell_type': 'code',
         'execution_count': None,
         'source': code,
         'metadata': {},
         'outputs': outputs})


def create_notebook(cells):
    """ Create a notebook with `cells`."""
    return nbformat.notebooknode.NotebookNode(
        {'cells': cells,
         'metadata': {},
         'nbformat': 4,
         'nbformat_minor': 4,
         })


def rm_first_line(text):
    """ Remove the first line in `text`."""
    return '\n'.join(text.split('\n')[1:])


# For the first cell of the notebook
INSTALL_CODE = """# Transformers installation
! pip install transformers
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git
"""


def convert_rst_file_to_notebook(
        rst_file,
        notebook_fname,
        framework=None,
        img_prefix=None,
        origin_folder=None,
        dest_folder=None
):
    r"""
    Convert rst_file to a notebook named notebook_fname.

    Args:
        - rst_file (:obj:`str`):
            The doc file to convert (in rst format).
        - notebook_fname (:obj:`str`):
            The output notebook file name (will be replaced if it exists).
        - framework (:obj:`str`, `optional`):
            If provided, must be :obj:`"pt"` or :obj:`"tf"`. In this case, only the PyTorch (resp. TensorFlow) version
            of the code is kept.
        - img_prefix (:obj:`str`, `optional`):
            If provided, will be inserted at the beginning of each image filename (in the `pytorch` or `tensorflow`
            folder, we need to add ../ to each image file to find them).
        - origin_folder (:obj:`str`, `optional`):
            If provided in conjunction with :obj:`dest_folder`, images encountered will be copied from this folder to
            :obj:`dest_folder`.
        - dest_folder (:obj:`str`, `optional`):
            If provided in conjunction with :obj:`origin_folder`, images encountered will be copied from
            :obj:`origin_folder` to this folder.
    """
    with open(rst_file, 'r') as f:
        content = f.read()
    lines = content.split("\n")
    lines = process_titles(lines)
    blocks = split_blocks(lines)
    cells = [code_cell(INSTALL_CODE)]
    for block, block_type in blocks:
        if block_type == 'title' or block_type == 'prose':
            block = convert_table(convert_rst_formatting(convert_rst_links(block)))
            cells.append(markdown_cell(block))
        elif block_type == 'anchor':
            block = convert_anchor(block)
            cells.append(markdown_cell(block))
        elif block_type.startswith('code-block'):
            codes = split_frameworks(block)
            if framework == 'pt' and len(codes) > 1:
                codes = (rm_first_line(codes[0]),)
            elif framework == 'tf' and len(codes) > 1:
                codes = (rm_first_line(codes[1]),)
            for code in codes:
                source, output = parse_code_and_output(code)
                if block_type.endswith('bash'):
                    lines = source.split("\n")
                    new_lines = [line if line.startswith("#") else f"! {line}" for line in lines]
                    source = "\n".join(new_lines)
                cells.append(code_cell(source, output=output))
        elif block_type.startswith("image"):
            image_name = block_type[len("image "):]
            block = convert_image(
                image_name,
                block,
                pref=img_prefix,
                origin_folder=origin_folder,
                dest_folder=dest_folder
            )
            cells.append(markdown_cell(block))
        elif block_type == "math":
            block = convert_math(block)
            cells.append(markdown_cell(block))
        else:
            block = convert_rst_formatting(convert_rst_links(block))
            block = convert_to_note(block, block_type)
            cells.append(markdown_cell(block))

    notebook = create_notebook(cells)
    nbformat.write(notebook, notebook_fname, version=4)


def convert_all_tutorials(path_to_docs=None, path_to_dest=None):
    """ Convert all tutorials into notebooks."""
    path_to_docs = PATH_TO_DOCS if path_to_docs is None else path_to_docs
    path_to_dest = PATH_TO_DEST if path_to_dest is None else path_to_dest
    for folder in ["pytorch", "tensorflow"]:
        os.makedirs(os.path.join(path_to_dest, folder), exist_ok=True)
    for file in TUTORIAL_FILES:
        notebook_name = os.path.splitext(file)[0] + ".ipynb"
        doc_file = os.path.join(path_to_docs, file)
        notebook_file = os.path.join(path_to_dest, notebook_name)
        convert_rst_file_to_notebook(doc_file, notebook_file, origin_folder=path_to_docs, dest_folder=path_to_dest)
        for folder, framework in zip(["pytorch", "tensorflow"], ["pt", "tf"]):
            notebook_file = os.path.join(os.path.join(path_to_dest, folder), notebook_name)
            convert_rst_file_to_notebook(doc_file, notebook_file, framework=framework, img_prefix="..")


if __name__ == "__main__":
    convert_all_tutorials()