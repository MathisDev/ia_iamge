pip3 install virtualenv
python3 -m venv sd_env
source sd_env/bin/activate
pip3 install torch torchvision torchaudio diffusers transformers ftfy regex tqdm
echo "from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

prompt = "Une magnifique peinture de paysage au coucher du soleil"
image = pipe(prompt).images[0]

image.save("output.png")
" >> main.py
python3 main.py
