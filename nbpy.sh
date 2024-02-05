jupyter nbconvert --to python $1
python_file=`echo $1 | sed -e "s/.ipynb/.py/g"`
sbatch -p gpu --gres=gpu:4 ./job.sh python $python_file ${@:2}