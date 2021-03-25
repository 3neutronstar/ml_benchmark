apt-get update
apt-get install python-pip
echo 'Upgrade Complete ==========================='
conda install tensorboard opencv matplotlib numpy 
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch -c conda-forge
conda install -c conda-forge opencv