#!/bin/bash

git lfs install
git clone https://huggingface.co/t5-base

python3 ./utils/huggingface_t5_ckpt_convert.py \
        -saved_dir t5-base-ft/ \
        -in_file t5-base \
        -inference_tensor_para_size 1 \
        -weight_data_type fp16

python3 test.py
