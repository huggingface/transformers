#!/usr/bin/env python3
import sys

from transformers import ProphetNetTokenizer, XLMProphetNetTokenizer


name = sys.argv[1]

if "xprophetnet" in name:
    tok = XLMProphetNetTokenizer.from_pretrained("microsoft/" + name)
else:
    tok = ProphetNetTokenizer.from_pretrained("microsoft/" + name)

tok.save_pretrained("/home/patrick/hugging_face/microsoft/" + name)
