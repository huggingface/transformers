#!/bin/bash

for FILE in converted/*; do 
  model_name=`basename $FILE`
  transformers-cli repo create $model_name -y
  git clone https://huggingface.co/Helsinki-NLP/$model_name
  mv $FILE/* $model_name/
  cd $model_name
  git add . && git commit -m "initial commit" 
  git push
  cd ..
done
