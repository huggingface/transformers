#!/bin/bash

#Download the latest wikipedia dump and save it in the path given by the first argument

if [ $# -ne 2 ]; then
    echo "Usage: ${0} path-to-destination path-to-extractor-tools"
    exit
fi

dump_path="${1}"

output_path="${2}"

python3 -m wikiextractor.WikiExtractor -b 100M --processes 64 -o "${output_path}" "${dump_path}"