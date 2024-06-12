from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Utilisez le CPU, car il n'y a pas de support CUDA sur votre Mac

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

prompt = input("describle a image: ") # "Une magnifique peinture de paysage au coucher du soleil"
image = pipe(prompt).images[0]

image.save("output.png")
