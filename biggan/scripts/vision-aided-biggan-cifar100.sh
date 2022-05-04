#!/bin/bash

WEIGHTS_PATH='./weights/'

python -u train.py --experiment_name vision-aided-cifar-100_clip --DiffAugment translation,cutout \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 600 --warmup 2000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--which_train_fn vision_aided --cv input-clip-output-conv  \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02  --toggle_grads \
--ema --use_ema --ema_start 1000  --weights_root $WEIGHTS_PATH  \
--test_every 2000 --save_every 2000 --seed 0 

cp -r $WEIGHTS_PATH/vision-aided-cifar-100_clip $WEIGHTS_PATH/vision-aided-cifar-100_clip_dino

python -u train.py --experiment_name vision-aided-cifar-100_clip_dino --DiffAugment translation,cutout \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 800 --resume --load_weights best \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--which_train_fn vision_aided --cv input-clip+dino-output-conv+conv  \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02  --toggle_grads \
--ema --use_ema --ema_start 1000   \
--test_every 2000 --save_every 2000 --seed 0 

cp -r $WEIGHTS_PATH/vision-aided-cifar-100_clip_dino $WEIGHTS_PATH/vision-aided-cifar-100_clip_dino_swin

python -u train.py --experiment_name vision-aided-cifar-100_clip_dino_swin --DiffAugment translation,cutout \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 1000 --resume --load_weights best \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--which_train_fn vision_aided --cv input-clip+dino+swin-output-conv+conv+conv  \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02  --toggle_grads \
--ema --use_ema --ema_start 1000   \
--test_every 2000 --save_every 2000 --seed 0 

