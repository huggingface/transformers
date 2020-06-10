#!/usr/bin/env python3
import sys

file_to_treat = sys[1]

output_hidden_states_to_add_string_1 = "\t\toutput_hidden_states (:obj:`bool`, `optional`, defaults to `:obj:`None`):\n"
output_hidden_states_to_add_string_2 = "\t\t\t If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.\n"

with open(file_to_treat, "r") as f1, open(file_to_treat + ".copy", "w") as f2:
    lines = f1.readlines()
    new_lines = []
    check_for_add_start = True
    for line in lines:
        if line.strip() == '"""':
            check_for_add_start = True
        elif line.strip() != '':
            check_for_add_start = False
        elif check_for_add_start is True and '@add_start_docstrings(' in line.strip():
            line = ''
            while line != '"""\n':
                line = new_lines.pop()
            new_lines.append(output_hidden_states_to_add_string_1)
            new_lines.append(output_hidden_states_to_add_string_2)
            new_lines.append('"""\n"')
            new_lines.append('\n')
            new_lines.append('\n')
        new_lines.append(line)
    for line in new_lines:
        f2.write(line)
