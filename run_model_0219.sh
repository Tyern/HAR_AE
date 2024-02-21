ModelVal="4cnn_128 4cnn_64"
TrainLimitData="1000 5000"
ClassNum="5"

python_file="15.1_enc_classify_n_classes.py"

count=0
for model_name in $ModelVal; do
    for train_limit_data in $TrainLimitData; do
        for class_num in $ClassNum; do
            ./sbgpu.sh $python_file -tl $train_limit_data --class_num $class_num --model_name $model_name
            echo $count $python_file -tl $train_limit_data --class_num $class_num --model_name $model_name
            count=`expr $count + 1`
        done
    done
done