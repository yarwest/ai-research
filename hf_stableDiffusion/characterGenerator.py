import argparse
import logging
import random
import sys
from promptBuilder import getCharacters, getMediumPrompts, getNegativeMedPrompts
from image import main as img

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", help="Character prompt", type=str, required=False, default="")
    parser.add_argument("--medium", help="Medium prompt collection", type=str, required=False, default="")
    return parser.parse_args()

def main(args):
    mediumPrompts = getMediumPrompts()
    characters = [ args.character] if args.character else getCharacters()
    negativePrompts = getNegativeMedPrompts()
    random.shuffle(characters)
    for char in characters:
        if args.medium:
            medium = args.medium
        else:
            medium = random.choice(list(mediumPrompts.keys()))
        prompts = mediumPrompts.get(medium)
        negativePrompt= negativePrompts.get(medium)
        logging.info(f"==== Generating {char} as {medium} ====")
        img_args = argparse.Namespace(
            prompt=f"{char},{','.join(prompts)}",
            negative_prompt=','.join(negativePrompt),
            batch_size=5,
            seed=1520,
            count=1,
            img_w=512,
            img_h=512,
            num_inference_steps=15,
            output_file=f"{char}-{medium}",
            base_img=None,
            mask_img=None,
            strict_mask=False,
            variation=False,
        )
        img(img_args)    

if __name__ == "__main__":
    main(parse_args())