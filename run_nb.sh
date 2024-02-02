jupyter nbconvert --to python $1
python_file=`echo $1 | sed -e "s/.ipynb/.py/g"`
sbatch -o .slurm/out.%a.out -p gpu --gres=gpu:4 ./job.sh python $python_file ${@:2}
sleep 1.5
rm $python_file