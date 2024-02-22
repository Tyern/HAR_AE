Method="random sampling"
SamplingHeuristic="uncertainty entropy margin"
SamplingSize="100"
# SamplingSize="50"
ResetWeight="1"
RandSeed="42 64"
# RandSeed="42"
python_file="16.3_AL_v4_500sampling.py"
ckpt_path="/nfs/ksdata/tran/HAR_AE/lightning_logs/10.3_classify/4cnn_64-8-100-1/version_0/checkpoints/sample_epoch=89-step=90-val_loss=1.038441.ckpt"
train_limit_data="500"

# Iterate the string variable using for loop
for reset_weight in $ResetWeight; do
    for randseed in $RandSeed; do
        for sampling_size in $SamplingSize; do
            ./sbgpu.sh $python_file --train_limit_data $train_limit_data --checkpoint_path $ckpt_path -m random --sampling_size $sampling_size --reset_weight $reset_weight --random_seed $randseed
            for sampling_heuristic in $SamplingHeuristic; do
                ./sbgpu.sh $python_file --train_limit_data $train_limit_data --checkpoint_path $ckpt_path -m sampling -sh $sampling_heuristic --sampling_size $sampling_size --reset_weight $reset_weight --random_seed $randseed
            done
        done
    done
done