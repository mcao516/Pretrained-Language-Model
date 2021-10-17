#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=MNLI
CLUSTER_NUM=3
FT_BERT_BASE_DIR=/home/mcao610/scratch/TinyBERT_TEST/${TASK_NAME}/teacher-cluster-${CLUSTER_NUM}
TMP_TINYBERT_DIR=/home/mcao610/scratch/TinyBERT_TEST/${TASK_NAME}/intermediate-cluster-${CLUSTER_NUM}-scratch
TASK_DIR=/home/mcao610/scratch/glue_data/${TASK_NAME}
TINYBERT_DIR=/home/mcao610/scratch/TinyBERT_TEST/${TASK_NAME}/final-cluster-${CLUSTER_NUM}-scratch-bs64

mkdir $TINYBERT_DIR
# --aug_train \
python /home/mcao610/Pretrained-Language-Model/TinyBERT/task_distill.py --pred_distill \
                       --teacher_model ${FT_BERT_BASE_DIR} \
                       --student_model ${TMP_TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${TINYBERT_DIR} \
                       --do_lower_case \
                       --learning_rate 3e-5 \
                       --num_train_epochs 3 \
                       --eval_step 1000 \
                       --max_seq_length 128 \
                       --train_batch_size 64 \
                       --k ${CLUSTER_NUM};
