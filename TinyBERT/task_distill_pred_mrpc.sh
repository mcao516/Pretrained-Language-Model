#!/bin/bash
source ~/env37/bin/activate

for SEQ in 4 8 16 32 64
do
    echo "SEQ: ${SEQ}"
    TASK_NAME=MRPC
    CLUSTER_NUM=2
    FT_BERT_BASE_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/teacher-${CLUSTER_NUM}
    TMP_TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/intermediate-2-seq${SEQ}
    TASK_DIR=$SCRATCH/glue_data/${TASK_NAME}
    TINYBERT_DIR=$SCRATCH/TinyBERT_TEST/${TASK_NAME}/final-${CLUSTER_NUM}-seq${SEQ}-aug-lr_1e6

    mkdir $TINYBERT_DIR
    python $HOME/Pretrained-Language-Model/TinyBERT/task_distill.py \
        --aug_train \
        --pred_distill \
        --teacher_model ${FT_BERT_BASE_DIR} \
        --student_model ${TMP_TINYBERT_DIR} \
        --data_dir ${TASK_DIR} \
        --task_name ${TASK_NAME} \
        --output_dir ${TINYBERT_DIR} \
        --do_lower_case \
        --learning_rate 1e-6 \
        --num_train_epochs 3 \
        --eval_step 300 \
        --max_seq_length 128 \
        --train_batch_size 128 \
        --k ${CLUSTER_NUM} \
        --cluster_map_path $HOME/Pretrained-Language-Model/TinyBERT/clusters/cluster_mrpc_k${CLUSTER_NUM}_aug.json;
done
