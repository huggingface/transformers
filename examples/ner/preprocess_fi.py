import sys
import pandas as pd
import os
import os.path as path
import math
import logging

logging.basicConfig(level=logging.INFO)

inp_file_path = sys.argv[1]
out_file_path = sys.argv[2]

def preprocess(inp_file, out_file):
    logging.info('Processing file {}'.format(inp_file))
    # read file
    input_df = pd.read_csv(inp_file, sep='\t', names=['TOKEN','ENTITY','NESTED_ENTITY'], skip_blank_lines=False, quoting=3)
    # remove tokens with unknown entity
    mask = (input_df['ENTITY'].notna() | (input_df['TOKEN'].isna() & input_df['ENTITY'].isna()))
    input_df = input_df.loc[mask]

    # iterate over rows and write to file
    stream = open(out_file, 'w')
    
    is_nan = True
    for i, row in enumerate(input_df.iterrows()):
        # if i > 200:
        #     break
        row = row[1]
        logging.debug(row['ENTITY'])
        if type(row['ENTITY']) != float:
            logging.debug('Writing to file')
            tok = row['TOKEN']
            stream.write('{} {}\n'.format(tok, row['ENTITY']))
            is_nan = False
        elif math.isnan(row['ENTITY']):
            if not is_nan:
                logging.debug('Adding newline')
                stream.write('\n')
                is_nan = True
        else:
            logging.warning('Detected ENTITY of type float not nan')

    stream.close()
    logging.info('Wrote to {}'.format(out_file))

preprocess(inp_file_path, out_file_path)
