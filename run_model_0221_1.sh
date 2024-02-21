ModelVal="4cnn_64"
TrainValTestLimitData="100 500"
ClassNum="8"
RadnSeed="1 2 3 4 5"

python_file="10.3_classify_nclasses.py"

count=0
for model_name in $ModelVal; do
    for train_val_test_limit_data in $TrainValTestLimitData; do
        for class_num in $ClassNum; do
            for random_seed in $RadnSeed; do
                ./sbgpu.sh $python_file --train_val_test_limit_data $train_val_test_limit_data --class_num $class_num --model_name $model_name --random_seed $random_seed
                echo $count $python_file --train_val_test_limit_data $train_val_test_limit_data --class_num $class_num --model_name $model_name --random_seed $random_seed
                count=`expr $count + 1`
            done
        done
    done
done