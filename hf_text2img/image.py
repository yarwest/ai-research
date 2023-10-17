import argparse
import logging
import sys
import os
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
)

import torch

__author__ = "Yarno Boelens"

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="Prompt for image generation", type=str, required=True)
    parser.add_argument("--base_img", help="Base image used for img2img diffusion", type=str, required=False)
    parser.add_argument("--data_dir", help="Directory with training data", type=str, required=False)
    parser.add_argument("--count", help="Number of generated images", type=str, required=False)
    parser.add_argument("--seed", help="Seed used for generation of random noise starting image", type=int, required=False, default=512)
    parser.add_argument("--num_inference_steps", help="The number of inference steps. If you want faster results you can use a smaller number. If you want potentially higher quality results, you can use larger numbers.", type=int, required=False, default=50)
    parser.add_argument("--img_w", help="Height of generated image", type=int, required=False, default=512)
    parser.add_argument("--img_h", help="Width of generated image", type=int, required=False, default=768)

    return parser.parse_args()

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('out/')

    num_images = args.count or 1
    prompt = [args.prompt] * num_images

    negativePrompt = ["ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"] * num_images

    # init pipelines & components
    text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                    #revision="fp16", torch_dtype=torch.float32
            )
    img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    generator = torch.manual_seed(args.seed)

    if args.base_img:
        base_image = Image.open(args.base_img).convert("RGB")
        base_image = base_image.resize((args.img_h, args.img_w))
        pipe = img2img(
            prompt=prompt,
            negative_prompt=negativePrompt,
            image=base_image,
            guidance_scale=7.5,
            height=args.img_h,
            width=args.img_w,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        )
    else:
        pipe = text2img(
            prompt,
            negative_prompt=negativePrompt,
            guidance_scale=7.5,
            height=args.img_h,
            width=args.img_w,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        )

    images = pipe.images

    fileName = '_'.join(args.prompt.split(' '))
    for i, image in enumerate(images):
        # save image with
        image.save(f"./out/{fileName}-{args.seed}-{args.num_inference_steps}its-{i}.png")


if __name__ == "__main__":
    main(parse_args())