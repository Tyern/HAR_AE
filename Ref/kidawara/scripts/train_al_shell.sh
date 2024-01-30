#! /root/bin/


python train_al.py --cuda --initial_model="results/ROLL_results/results_test1/Baseline/best_model.pth" --outdir="results/ROLL_results/results_test13/aUM" --initdir="results/ROLL_results/results_test13/Baseline" --train_car_ids=[5,10,11,13] --val_car_ids=[5,10,11,13] --test_car_ids=[1]
