python3 main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda:2 --num_layers=1 --num_heads=5
# CUDA_VISIBLE_DEVICES=5 python3 main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model bert --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda:3 --num_layers=1 --num_heads=5