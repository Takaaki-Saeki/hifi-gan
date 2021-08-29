source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load python/3.8/3.8.7
module load cuda/11.2/11.2.2
module load cudnn/8.2/8.2.0
module load nccl/2.8/2.8.4-1
source $HOME/convtasnet/bin/activate

localdir="${SGE_LOCALDIR}"
target_dataset_root="${localdir}/dataset"
rm -rf ${target_dataset_root}
echo "###### Copying data to ${localdir} ######"
mkdir -p ${target_dataset_root}
cp -r /$HOME/dataset/LJSpeech-1.1 ${target_dataset_root}/
cp -r LJSpeech-1.1 ${localdir}/
echo "###### done. ######"

pip install --upgrade pip
pip install -r requirements.txt

python train.py \
    --config config_v1.json \
    --input_wavs_dir ${target_dataset_root}/LJSpeech-1.1/wavs \
    --input_mels_dir ${target_dataset_root}/ft_dataset \
    --input_training_file ${localdir}/LJSpeech-1.1/training.txt \
    --input_validation_file ${localdir}/LJSpeech-1.1/validation.txt
