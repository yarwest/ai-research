import argparse
import logging
import sys
from promptBuilder import getCharacters, getMediumPrompts
from image import main as img

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main():
    mediumPrompts = getMediumPrompts()
    for char in getCharacters():
        for medium, prompts in mediumPrompts.items():
            logging.info(f"==== Generating {char} as {medium} ====")
            args = argparse.Namespace(
                prompt=f"{char},{','.join(prompts)}",
                batch_size=10,
                seed=152,
                count=1,
                img_w=512,
                img_h=512,
                num_inference_steps=10,
                output_file=f"{char}-{medium}",
                base_img=None,
                mask_img=None,
                strict_mask=False,
                variation=False,      
            )
            img(args)

if __name__ == "__main__":
    main()