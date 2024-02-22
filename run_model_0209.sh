# ModelVal="4cnnmp_128 4cnnmp_64 4cnn_128 4cnn_64 poor_128 poor_64 rich_128 rich_64"
ModelVal="4cnn_64"
ClassNum="8"
TrainLimit="1000"
RandSeed="65 66 67"

python_file="10.1_classify_nclasses.py"

# Iterate the string variable using for loop
for modelname in $ModelVal; do
    for classnum in $ClassNum; do
        for trainlim in $TrainLimit; do
            for randseed in $RandSeed; do
                ./sbgpu.sh $python_file -m $modelname -cn $classnum -tl $trainlim -r $randseed
            done
        done
    done
done