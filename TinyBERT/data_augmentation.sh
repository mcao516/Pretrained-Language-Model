#!/bin/bash
source ~/env37/bin/activate

BERT_BASE_DIR=/home/mcao610/scratch/huggingface/bert-base-uncased
GLOVE_EMB=/home/mcao610/scratch/GloVe/glove.6B.300d.txt
GLUE_DIR=/home/mcao610/scratch/glue_data/
TASK_NAME=SST-2

python data_augmentation.py --pretrained_bert_model $BERT_BASE_DIR \
                            --glove_embs $GLOVE_EMB \
                            --glue_dir $GLUE_DIR \
                            --task_name $TASK_NAME;
