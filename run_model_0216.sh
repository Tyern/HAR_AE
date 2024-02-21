SaveModelFolder="20_AL_train_unbalance/4cnn_64-8-1000-3-10-300-64 20_AL_train_unbalance/4cnn_64-8-1000-3-20-300-42 20_AL_train_unbalance/4cnn_64-8-1000-3-30-400-64 20_AL_train_unbalance/4cnn_64-8-1000-3-50-200-64 20_AL_train_unbalance/4cnn_64-8-1000-3-100-400-42 20_AL_train_unbalance/4cnn_64-8-1000-4-10-400-64 20_AL_train_unbalance/4cnn_64-8-1000-4-20-300-64 20_AL_train_unbalance/4cnn_64-8-1000-4-30-200-64 20_AL_train_unbalance/4cnn_64-8-1000-4-50-400-42 20_AL_train_unbalance/4cnn_64-8-1000-4-100-300-64 20_AL_train_unbalance/4cnn_64-8-1000-5-10-400-64 20_AL_train_unbalance/4cnn_64-8-1000-5-20-200-42 20_AL_train_unbalance/4cnn_64-8-1000-5-30-300-64 20_AL_train_unbalance/4cnn_64-8-1000-5-50-400-42 20_AL_train_unbalance/4cnn_64-8-1000-5-100-300-64"

Method="sampling random"
SamplingSize="10 20 30 50 100"
ResetWeight="1 0"
RandSeed="64"

python_file="20.2_AL_train_unbalance.py"

count=0
for randseed in $RandSeed; do
    for method in $Method; do
        for sampling_size in $SamplingSize; do
            for reset_weight in $ResetWeight; do
                for save_model in $SaveModelFolder; do
                    ./sbgpu.sh $python_file --save_model_folder $save_model --random_seed $randseed --method $method --sampling_size $sampling_size --reset_weight $reset_weight
                    echo $count --save_model_folder $save_model --random_seed $randseed --method $method --sampling_size $sampling_size --reset_weight $reset_weight
                    count=`expr $count + 1`
                done
            done
        done
    done
done