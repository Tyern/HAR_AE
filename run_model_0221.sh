ModelVal="4cnn_128 4cnn_64"
TrainLimitData="1000 5000"
ClassNum="8"
DataMergeParam="1"

python_file="10.2_classify_nclasses.py"

count=0
for model_name in $ModelVal; do
    for train_limit_data in $TrainLimitData; do
        for class_num in $ClassNum; do
            for data_merge_param in $DataMergeParam; do
                ./sbgpu.sh $python_file -tl $train_limit_data --class_num $class_num --model_name $model_name --data_merge_param $data_merge_param
                echo $count $python_file -tl $train_limit_data --class_num $class_num --model_name $model_name --data_merge_param $data_merge_param
                count=`expr $count + 1`
            done
        done
    done
done