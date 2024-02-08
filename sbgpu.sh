python_file=$1
sbatch -p gpu --gres=gpu:4 ./job.sh python $python_file ${@:2}