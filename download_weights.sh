echo "=== Downloading BERT pre-trained weights ==="
echo "---"
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz
rm -rf simple-examples.tgz
