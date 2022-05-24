#!/usr/bin/env bash
doc_file=${1}

python utils/prepare_for_doc_test.py src/transformers/utils/doc.py ${doc_file}
pytest --doctest-modules ${doc_file} -sv --doctest-continue-on-failure
#python utils/prepare_for_doc_test.py src docs --remove_new_line