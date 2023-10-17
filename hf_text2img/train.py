import argparse
import logging
import sys
import os

from diffusers import StableDiffusionPipeline
import torch

__author__ = "Yarno Boelens"


np.set_printoptions(suppress=True)


# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Directory with training data", type=str, required=False)
    parser.add_argument("--prompt", help="Prompt for image generation", type=str, required=True)
    parser.add_argument("--count", help="Number of generated images", type=str, required=False)
    return parser.parse_args()

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('./out/')

    num_images = args.count or 1
    prompt = [args.prompt] * num_images

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
    generator = torch.Generator("cuda").manual_seed(1024)
    images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images

    fileName = '_'.join(args.prompt.split(' '))
    for image, i in images:
        # save image with
        image.save(f"./out/{fileName}{i}.png")


if __name__ == "__main__":
    main(parse_args())