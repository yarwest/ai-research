import argparse
import logging
import sys


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
    parser.add_argument("--seed", help="Seed used for generation of random noise starting image", type=int, required=False, default=512)
    parser.add_argument("--num_inference_steps", help="The number of inference steps. If you want faster results you can use a smaller number. If you want potentially higher quality results, you can use larger numbers.", type=int, required=False, default=50)

    return parser.parse_args()

def main(args):
    negativePrompts = [
        "ugly",
        "tiling",
        "poorly drawn hands",
        "poorly drawn feet",
        "poorly drawn face",
        "poorly drawn eyes",
        "poorly drawn teeth",
        "out of frame",
        "mutation",
        "mutated",
        "extra limbs",
        "extra legs",
        "extra arms",
        "extra fingers",
        "disfigured",
        "deformed",
        "cross-eye",
        "body out of frame",
        "head out of frame",
        "blurry",
        "bad art",
        "bad anatomy",
        "blurred",
        "text",
        "watermark",
        "grainy",
        "underexposed",
        "overexposed",
        "beginner",
        "amateur",
        "distorted face",
        "signature",
        "low contrast",
        "cut off"
    ]

    goodPrompts = [
        "cinematic lighting",
        "4k",
        "futuristic",
        "sci-fi",
        "neon lightning",
        "photorealistic",
        "highly detailed", # !!
        "intricate design", # !!
        "sharp focus",
        "dark",
        "dystopian",
        "stunningly beautiful",
        "dslr",
        "ray tracing",
        "realistic",
    ]

    prompt = args.prompt

    for i in range(0,goodPrompts.__len__):
        for j in range(0,goodPrompts.__len__):
            if i != j:
                for x in range(0,goodPrompts.__len__):
                    if x != i and x != j:
                         text2img(
                            prompt=[
                                args.prompt, goodPrompts[i], goodPrompts[j], goodPrompts[x]
                            ],
                            negativePrompts=[negativePrompts.join(',')])



if __name__ == "__main__":
    main(parse_args())