#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=MNLI
CLUSTER_NUM=3
FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-cluster-${CLUSTER_NUM}
GENERAL_TINYBERT_DIR=$SCRATCH/General_TinyBERT_6L_768D
TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate-cluster-${CLUSTER_NUM}-scratch

mkdir $TMP_TINYBERT_DIR
python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
    --teacher_model ${FT_BERT_BASE_DIR} \
    --student_model ${GENERAL_TINYBERT_DIR} \
    --data_dir ${TASK_DIR} \
    --task_name ${TASK_NAME} \
    --output_dir ${TMP_TINYBERT_DIR} \
    --max_seq_length 128 \
    --train_batch_size 64 \
    --num_train_epochs 10 \
    --eval_step 3000 \
    --aug_train \
    --do_lower_case \
    --k ${CLUSTER_NUM} \
    --init_student_from_scratch;