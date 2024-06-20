#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
    --config_file accelerate_config.yaml \
train_dpo.py \
    --tokenizer_name_or_path ${CONFIG_NAME} \
    --policy_model_path ${INIT_ACTOR_PATH} \
    --adv_model_path ${INIT_ADV_PATH} \
    --critic_model_path ${INIT_REWARD_PATH} \
    --llama_guard_path ${INIT_GUARD_PATH} \
    --similarity_model_path ${INIT_SIMILARITY_PATH} \
    --model_save_path outputs/models/gpo/gpo_model \
    --data_path data/ppo_data \
    --seed 42 \
    --maxlen_prompt 2048 \
    --maxlen_res 512 \
    --lr 5e-7 \
    --critic_lr 1.5e-6 \
    --gamma 1. \
    --lam 0.95 \
    --entropy_clip 35.0 \
    --value_clip 0.2 \
    --pg_clip 0.2 \
    --reward_clip 0. \
    --entropy_loss_weight 0. \
    --ppo_pretrain_loss_weight 0. \
    --bleu_weight 5. \
    --embed_weight 1. \
    --kl_penalty_weight 0.01 \
    --train_steps 800 \
    --save_per_step 100 \
    --warmup_steps 100 \
    --defend_stage_steps 400 \
    --attack_stage_steps 200 \
    --batch_size 8 \
    --rollout_batch_size 8 \
    --num_rollouts 8 \
    --gradient_checkpoint \
    --lang en \
    --logdir outputs/tensorboard_log/gpo/gpo_model \
&> outputs/log/ppo_model.log
