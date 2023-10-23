import argparse
import logging
import sys
import os
import torch
import numpy as np 
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionImageVariationPipeline
)

from promptBuilder import getNegativePrompts, getPositivePrompts

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
    parser.add_argument("--prompt", help="Prompt for image generation", type=str, required=False, default="")
    parser.add_argument("--base_img", help="Relative path to base image used for img2img diffusion", type=str, required=False)
    parser.add_argument("--mask_img", help="Relative path to mask image used for inpainting. White pixels in the mask are repainted while black pixels are preserved", type=str, required=False)
    parser.add_argument("--strict_mask", help="Pass True to use strict masking during inpainting, to ensure preservation of the black pixels from the mask image", type=bool, required=False, default=False)
    parser.add_argument("--data_dir", help="Directory with training data", type=str, required=False)
    parser.add_argument("--variation", help="Pass True to use the Image Variation model", type=bool, required=False)
    parser.add_argument("--count", help="Number of generated images", type=str, required=False, default=1)
    parser.add_argument("--seed", help="Seed used for generation of random noise starting image", type=int, required=False, default=512)
    parser.add_argument("--num_inference_steps", help="The number of inference steps. If you want faster results you can use a smaller number. If you want potentially higher quality results, you can use larger numbers.", type=int, required=False, default=50)
    parser.add_argument("--img_w", help="Height of generated image", type=int, required=False, default=512)
    parser.add_argument("--img_h", help="Width of generated image", type=int, required=False, default=512)

    return parser.parse_args()

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('out/')

    prompt = [args.prompt] * args.count

    negativePrompt = [getNegativePrompts()] * args.count

    # init pipelines & components
    text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                    #revision="fp16", torch_dtype=torch.float32
            )

    generator = torch.manual_seed(args.seed)

    if args.base_img:
        base_image = Image.open(os.path.join(os.path.dirname(__file__), args.base_img)).convert("RGB")
        if(base_image.width != args.img_w or base_image.height != args.img_h):
            base_image = base_image.resize((args.img_w, args.img_h))
        if(args.mask_img):
            inpainting = StableDiffusionInpaintPipeline(**text2img.components).from_pretrained("runwayml/stable-diffusion-inpainting")
            mask_image = Image.open(os.path.join(os.path.dirname(__file__), args.mask_img)).convert("L")
            if(mask_image.width != args.img_w or mask_image.height != args.img_h):
                mask_image = mask_image.resize((args.img_w, args.img_h))
            
            pipe = inpainting(
                prompt=f"{args.prompt}",
                negative_prompt=getNegativePrompts(),
                image=base_image,
                mask_image=mask_image,
                guidance_scale=7.5,
                strength=0.75,
                num_images_per_prompt=args.count,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                height=args.img_h,
                width=args.img_w
            )
        elif(args.variation):
            variation = StableDiffusionImageVariationPipeline.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
            )
            pipe = variation(
                guidance_scale=7.5,
                height=args.img_h,
                width=args.img_w,
                image=base_image,
                num_inference_steps=args.num_inference_steps,
                generator=generator
            )
        else:
            img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
            pipe = img2img(
                prompt=prompt,
                negative_prompt=negativePrompt,
                image=base_image,
                guidance_scale=7.5,
                strength=0.75,
                num_inference_steps=args.num_inference_steps,
                generator=generator
            )
    else:
        pipe = text2img(
            prompt=prompt,
            negative_prompt=negativePrompt,
            guidance_scale=7.5,
            height=args.img_h,
            width=args.img_w,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        )

    images = pipe.images

    fileName = '_'.join(args.prompt.split(' '))
    if(args.base_img):
        if(args.mask_img):
            fileName += '-inpainting'
        elif(args.variation):
            fileName += '-variation'
        else:
            fileName += '-img2img'
    for i, image in enumerate(images):
        if(args.mask_img and args.strict_mask):
            # Convert mask to grayscale NumPy array
            mask_image_arr = np.array(mask_image.convert("L"))
            # Add a channel dimension to the end of the grayscale mask
            mask_image_arr = mask_image_arr[:, :, None]
            # Binarize the mask: 1s correspond to the pixels which are repainted
            mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
            mask_image_arr[mask_image_arr < 0.5] = 0
            mask_image_arr[mask_image_arr >= 0.5] = 1
            # Take the masked pixels from the repainted image and the unmasked pixels from the initial image
            unmasked_unchanged_image_arr = (1 - mask_image_arr) * base_image + mask_image_arr * image
            image = Image.fromarray(unmasked_unchanged_image_arr.round().astype("uint8"))

        # save image with
        image.save(f"./out/{fileName}-{args.seed}-{args.num_inference_steps}its-{i}.png")


if __name__ == "__main__":
    main(parse_args())