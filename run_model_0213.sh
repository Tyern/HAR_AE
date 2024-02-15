Method="random sampling full"
SamplingHeuristic="uncertainty entropy margin"
SamplingSize="50 100 200 300 500 1000"
# SamplingSize="50"
ResetWeight="0 1"
RandSeed="42 64 65 66 67"
# RandSeed="42"

# Iterate the string variable using for loop
for reset_weight in $ResetWeight; do
    for randseed in $RandSeed; do
        ./sbgpu.sh $1 -m full --reset_weight $reset_weight --random_seed $randseed
        for sampling_size in $SamplingSize; do
            ./sbgpu.sh $1 -m random --sampling_size $sampling_size --reset_weight $reset_weight --random_seed $randseed
            for sampling_heuristic in $SamplingHeuristic; do
                ./sbgpu.sh $1 -m sampling -sh $sampling_heuristic --sampling_size $sampling_size --reset_weight $reset_weight --random_seed $randseed
            done
        done
    done
done