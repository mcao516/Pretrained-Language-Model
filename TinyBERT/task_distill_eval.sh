#!/bin/bash
source ~/env37/bin/activate

TASK_NAME=MNLI
TINYBERT_DIR=/home/mcao610/scratch/TinyBERT_TEST/${TASK_NAME}/final-similarity
# TINYBERT_DIR=/home/mcao610/scratch/6L_768D_FinalModel/MNLI
TASK_DIR=/home/mcao610/scratch/glue_data/${TASK_NAME}
OUTPUT_DIR=/home/mcao610/scratch/TinyBERT_TEST/${TASK_NAME}/eval  # output directory

mkdir $OUTPUT_DIR
python task_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${OUTPUT_DIR} \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128;