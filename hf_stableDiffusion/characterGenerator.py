import argparse
from promptBuilder import getCharacters, getMediumPrompts
from image import main as img

def main():
    mediumPrompts = getMediumPrompts()
    for char in getCharacters():
        for medium, prompts in mediumPrompts.items():
            args = argparse.Namespace(
                prompt=f"{char},{','.join(prompts)}",
                batch_size=10,
                seed=152,
                count=1,
                img_w=512,
                img_h=512,
                num_inference_steps=10,
                output_file=f"{char}-{medium}"
            )
            img(args)

if __name__ == "__main__":
    main()