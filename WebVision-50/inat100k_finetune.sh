CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_webvision_Sel-CL_fine-tuning.py --epoch 50 --num_classes 1000 --batch_size 64 \
--network "PARN18" --lr 0.001 --wd 1e-4 --dataset "inat100k" --alpha_m 1.0 --seed_initialization 1 --seed_dataset 42 \
--headType "Linear" --ReInitializeClassif 1 --startLabelCorrection 20 --experiment_name inat100k_finetune \
--out /home/trode/selcl/out/ --wandb-project inaturalist
