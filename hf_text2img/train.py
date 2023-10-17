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
    return parser.parse_args()

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('./out/')

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
    
    image = pipe(args.prompt).images[0]
    fileName = '_'.join(args.prompt.split(' '))
    # save image with
    image.save(f"./out/{fileName}.png")


if __name__ == "__main__":
    main(parse_args())