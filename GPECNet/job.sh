!/bin/bash 
#PBS -q week
#PBS -o out.o
#PBS -e out.e
#PBS -N exp
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -V

cd ${PBS_O_WORKDIR}
echo "Running on: " 
cat ${PBS_NODEFILE}
cat $PBS_NODEFILE > machines.list
echo "Program Output begins: " 
source /users/home/amit_unde/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_cpu_only
python /wd/users/amit_unde/PECNet-master-next_sigma_gpu/training_loop.py -cfn optimal.yaml -sf /wd/users/amit_unde/PECNet-master-next_sigma_gpu/saved_models/abc.pt        
#python /wd/users/amit_unde/original_track_rcnn/main.py configs/conv3d_sep2 "{\"KITTI_segtrack_data_dir\": \"/wd/users/amit_unde/sigmoid_learning_every_epoch/data/KITTI_MOTS/train/\", \"batch_size\":4,\"max_saves_to_keep\":5,\"learning_rates\":\"{1: 0.0000005}\",\"model\":\"conv3d_sep_r1\",\"load_init\":\"models/converted\"}"         


