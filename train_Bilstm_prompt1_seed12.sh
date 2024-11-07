#!/usr/bin/env bash
model_name='ProTACT_Bilstm'
prompt=1
seed=12
python train_Bilstm_ProTACT.py --test_prompt_id ${prompt} --model_name ${model_name} --seed ${seed} --num_heads 2 --features_path 'data/LDA/hand_crafted_final_'
