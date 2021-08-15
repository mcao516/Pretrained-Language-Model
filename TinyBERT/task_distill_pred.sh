#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=MNLI
FT_BERT_BASE_DIR=/home/mcao610/scratch/huggingface/MNLI/uncased/
TMP_TINYBERT_DIR=/home/mcao610/scratch/TinyBERT_TEST/MNLI/intermediate
TASK_DIR=/home/mcao610/scratch/glue_data/${TASK_NAME}
TINYBERT_DIR=/home/mcao610/scratch/TinyBERT_TEST/${TASK_NAME}/final  # output directory

mkdir $TINYBERT_DIR
python task_distill.py --pred_distill \
                       --teacher_model ${FT_BERT_BASE_DIR} \
                       --student_model ${TMP_TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${TINYBERT_DIR} \
                       --aug_train \
                       --do_lower_case \
                       --learning_rate 3e-5 \
                       --num_train_epochs 3 \
                       --eval_step 300 \
                       --max_seq_length 128 \
                       --train_batch_size 32;