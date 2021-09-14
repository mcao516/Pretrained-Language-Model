#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=MNLI
CLUSTER_NUM=2
BERT_BASE_DIR=/home/mcao610/scratch/huggingface/bert-base-uncased
TASK_DIR=/home/mcao610/scratch/glue_data/${TASK_NAME}
OUTPUT_DIR=/home/mcao610/scratch/TinyBERT_TEST/${TASK_NAME}/teacher-cluster-${CLUSTER_NUM} # output directory

mkdir $OUTPUT_DIR
CUDA_VISIBLE_DEVICE=0,1,2,3 python /home/mcao610/Pretrained-Language-Model/TinyBERT/train_teacher.py \
                            --teacher_model ${BERT_BASE_DIR} \
                            --data_dir ${TASK_DIR} \
                            --task_name ${TASK_NAME} \
                            --output_dir ${OUTPUT_DIR} \
                            --do_lower_case \
                            --learning_rate 3e-5 \
                            --num_train_epochs 3 \
                            --eval_step 500 \
                            --max_seq_length 128 \
                            --train_batch_size 32 \
                            --aug_train \
                            --k ${CLUSTER_NUM};