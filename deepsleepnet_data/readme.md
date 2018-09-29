The files in this folders were copied with minor modifications from #Source : https://github.com/akaraspt/deepsleepnet

#To get the dataset :

cd data
chmod +x download_physionet.sh
./download_physionet.sh

###Those scripts taken from the deepsleepnet only work with python2
python2 prepare_physionet.py --data_dir data --output_dir data/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'