#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=MNLI
CLUSTER_NUM=3
BERT_BASE_DIR=$SCRATCH/huggingface/bert-base-uncased
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
OUTPUT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-cluster-${CLUSTER_NUM}-scratch # output directory

mkdir $OUTPUT_DIR
#                                 --aug_train \
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
                                --train_batch_size 128 \
                                --k ${CLUSTER_NUM} \
                                --init_from_scratch;
