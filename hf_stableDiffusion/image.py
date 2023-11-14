import argparse
import logging
import sys
import os
import torch
import numpy as np
import uuid
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
    parser.add_argument("--negative_prompt", help="Negative prompt for image generation", type=str, required=False, default="")
    parser.add_argument("--base_img", help="Relative path to base image used for img2img diffusion", type=str, required=False)
    parser.add_argument("--mask_img", help="Relative path to mask image used for inpainting", type=str, required=False)
    parser.add_argument("--strict_mask", help="Pass True to use strict masking during inpainting", type=bool, required=False, default=False)
    parser.add_argument("--data_dir", help="Directory with training data", type=str, required=False)
    parser.add_argument("--variation", help="Pass True to use the Image Variation model", type=bool, required=False)
    parser.add_argument("--batch_size", help="Number of generators to be used in deterministic batch generation", type=int, required=False, default=1)
    parser.add_argument("--count", help="Number of generated images", type=int, required=False, default=1)
    parser.add_argument("--seed", help="Seed used for generation of random noise starting image", type=int, required=False, default=512)
    parser.add_argument("--num_inference_steps", help="The number of inference steps. If you want faster results you can use a smaller number. If you want potentially higher quality results, you can use larger numbers.", type=int, required=False, default=50)
    parser.add_argument("--img_w", help="Height of generated image", type=int, required=False, default=512)
    parser.add_argument("--img_h", help="Width of generated image", type=int, required=False, default=512)
    parser.add_argument("--output_file", help="String to include in outputfile name", type=str, required=False)

    return parser.parse_args()

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('out/')

    processID = uuid.uuid4()
    print("---- Process ID:", processID)

    prompt = [args.prompt] * args.count

    if args.negative_prompt:
        negativePrompt = [args.negative_prompt] * args.count
    else:
        negativePrompt = [getNegativePrompts()] * args.count

    logging.info("==== Initialising HuggingFace Stable Diffusion pipelines ====")
    # init pipelines & components
    text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                    #revision="fp16", torch_dtype=torch.float32
            )

    if(args.batch_size > 1):
        generator = [torch.manual_seed(args.seed - i) for i in range(args.batch_size)]
        nImages = args.batch_size
    else:
        generator = [torch.manual_seed(args.seed) for i in range(args.count)]
        nImages = args.count

    if args.base_img:
        base_image = Image.open(os.path.join(os.path.dirname(__file__), args.base_img)).convert("RGB")
        if(base_image.width != args.img_w or base_image.height != args.img_h):
            base_image = base_image.resize((args.img_w, args.img_h))
        if(args.mask_img):
            inpainting = StableDiffusionInpaintPipeline(**text2img.components).from_pretrained("runwayml/stable-diffusion-inpainting")
            mask_image = Image.open(os.path.join(os.path.dirname(__file__), args.mask_img)).convert("L")
            if(mask_image.width != args.img_w or mask_image.height != args.img_h):
                mask_image = mask_image.resize((args.img_w, args.img_h))
            
            logging.info("==== Starting inpainting pipeline ====")
            pipe = inpainting(
                prompt=f"{args.prompt}",
                negative_prompt=getNegativePrompts(),
                image=base_image,
                mask_image=mask_image,
                guidance_scale=7.5,
                strength=0.75,
                num_images_per_prompt=nImages,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                height=args.img_h,
                width=args.img_w
            )
        elif(args.variation):
            variation = StableDiffusionImageVariationPipeline.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
            )
            logging.info("==== Starting variation pipeline ====")
            pipe = variation(
                guidance_scale=7.5,
                height=args.img_h,
                width=args.img_w,
                image=base_image,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                num_images_per_prompt=nImages
            )
        else:
            img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
            logging.info("==== Starting img2img pipeline ====")
            pipe = img2img(
                prompt=prompt,
                negative_prompt=negativePrompt,
                image=base_image,
                guidance_scale=7.5,
                strength=0.75,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                num_images_per_prompt=nImages
            )
    else:
        logging.info("==== Starting text2img pipeline ====")
        pipe = text2img(
            prompt=prompt,
            negative_prompt=negativePrompt,
            guidance_scale=7.5,
            height=args.img_h,
            width=args.img_w,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            num_images_per_prompt=nImages
        )

    logging.info("==== Pipeline completed, storing images ====")
    images = pipe.images

    fileName = 'text2img'
    if(args.base_img):
        if(args.mask_img):
            fileName = 'inpainting'
        elif(args.variation):
            fileName = 'variation'
        else:
            fileName = 'img2img'
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
        image.save(f"./out/{processID}-{f'-{args.output_file}' if args.output_file else ''}{i}.png")

    with open('output-log.txt', 'a+') as file:
        if(os.stat("output-log.txt").st_size == 0):
            file.write(f"process | algorithm | prompt | seed | num_inference_steps | number\n")
        file.write(f"{processID}|{fileName}|{args.prompt}|{args.seed}|{args.num_inference_steps}its|{i}\n")

if __name__ == "__main__":
    main(parse_args())