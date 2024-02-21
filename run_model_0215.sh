TargetClassDataNum="10 20 30 50 100"
RandSeed="42 64"
Patience="200 300 400"
TargetClass="3 4 5"

python_file="20_AL_train_unbalance.py"

# Iterate the string variable using for loop
for target_class_data_num in $TargetClassDataNum; do
    for patience in $Patience; do
        for randseed in $RandSeed; do
            for target_class in $TargetClass; do
                ./sbgpu.sh $python_file --target_class_data_num $target_class_data_num --random_seed $randseed --patience $patience --target_class $target_class
            done
        done
    done
done