import argparse
import logging
import sys
import os

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

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
    parser.add_argument("--prompt", help="Prompt for video generation", type=str, required=True)
    parser.add_argument("--base_img", help="Base video used for video2video diffusion", type=str, required=False)
    parser.add_argument("--data_dir", help="Directory with training data", type=str, required=False)
    parser.add_argument("--count", help="Number of generated videos", type=str, required=False)
    parser.add_argument("--seed", help="Seed used for generation of random noise starting image", type=int, required=False, default=512)
    parser.add_argument("--num_inference_steps", help="The number of inference steps. If you want faster results you can use a smaller number. If you want potentially higher quality results, you can use larger numbers.", type=int, required=False, default=50)
    parser.add_argument("--img_w", help="Height of generated video", type=int, required=False, default=512)
    parser.add_argument("--img_h", help="Width of generated video", type=int, required=False, default=768)

    return parser.parse_args()

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('out/')

    num_videos = args.count or 1
    prompt = [args.prompt] * num_videos

    negativePrompt = ["ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"] * num_videos

    # init pipelines & components
    text2video = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b"
                                                 #, torch_dtype=torch.float16, variant="fp16"
    )
    #video2video = StableDiffusionImg2ImgPipeline(**text2video.components)
    generator = torch.manual_seed(args.seed)

    if args.base_video:
        base_image = Image.open(args.base_img).convert("RGB")
        base_image = base_image.resize((args.img_h, args.img_w))
        #video2video
    else:
        pipe = text2video(
            prompt,
            negative_prompt=negativePrompt,
            guidance_scale=7.5,
            height=args.img_h,
            width=args.img_w,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        
    video_frames = pipe.frames

    fileName = '_'.join(args.prompt.split(' '))
    video_path = export_to_video(video_frames, output_video_path=f"./out/{fileName}-{args.seed}-{args.num_inference_steps}its.mp4")

if __name__ == "__main__":
    main(parse_args())