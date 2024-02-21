# ModelVal="4cnnmp_128 4cnnmp_64 4cnn_128 4cnn_64 poor_128 poor_64 rich_128 rich_64"
ModelVal="4cnnmp_128 4cnnmp_64 4cnn_128 4cnn_64"
ClassNum="5"
TrainLimit="5000"
RandSeed="42 64"

# Iterate the string variable using for loop
for modelname in $ModelVal; do
    for classnum in $ClassNum; do
        for trainlim in $TrainLimit; do
            for randseed in $RandSeed; do
                ./sbgpu.sh $1 -m $modelname -cn $classnum -tl $trainlim -r $randseed
            done
        done
    done
done