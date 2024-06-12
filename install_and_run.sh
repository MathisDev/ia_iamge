pip3 install virtualenv
python3 -m venv sd_env
source sd_env/bin/activate
pip3 install torch torchvision torchaudio diffusers transformers ftfy regex tqdm
python3 main.py
