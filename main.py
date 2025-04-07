from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"  # public + same model as sd-legacy
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)  # use float32 on CPU
pipe = pipe.to("cpu")  # run on CPU

prompt = input("IMAGE: ")
image = pipe(prompt, num_inference_steps=3).images[0]

image.save(f'{prompt.replace(" ", "")}.png')