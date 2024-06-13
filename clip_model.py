from diffusers import StableDiffusionPipeline

#model_id = "CompVis/stable-diffusion-v1-4"
model_id= "path/to/clip-guided-diffusion-model"
device = "cpu"  # Utilisez le CPU, car il n'y a pas de support CUDA sur votre Mac

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

prompt = input("describle a image: ")

image = pipe(prompt).images[0]
print("Le nombre d'image generer :"len(image.images))
image.save("output.png")
