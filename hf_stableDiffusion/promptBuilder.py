import argparse
import logging
import sys
from image import main as img

__author__ = "Yarno Boelens"

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def getNegativePrompts():
    return ",".join([
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
        "low resolution",
        "low details",
        "duplicate artifacts",
        "cut off"
    ])

def getPositivePrompts():
    return ",".join([
        "photo",
        "photography",
        "dslr",
        "sharp focus",
        "futuristic",
        "sci-fi",
        "neon lightning",
        "photorealistic",
        "high resolution", # !!
        "highly detailed", # !!
        "intricate design", # !!
        "dark",
        "dystopian",
        "stunningly beautiful",
        "ray tracing",
        "realistic",
        "national geographic",
        "cinematic lighting",
        "4k",
    ])

def getArtstyles():
    return [
        "pop art",
        "abstract",
        "impressionist",
        "surrealist",
        "baroque-style",
        "cubist",
        "art nouveau",
        "gothic",
        "cyberpunk",
        "pixel art",
        "isometric",
        "renaissance"
    ]

def getCharacters():
    return [
        "batman",
        "joker",
        "superman",
        "wolverine",
        "deadpool",
        "yoda",
        "darth vader",
        "jedi",
        "geralt of rivia",
        "freddy krueger",
        "sonic",
        "mario",
        "yoshi",
        "gandalf and hobbits",
        "iron man",
        "harry potter",
        "harry potter vs voldemort",
        "optimus prime, transformers",
        "bumblebee, transformers",
        "spiderman",
        "neo, matrix",
        "sherlock holmes",
        "minions",
        "skyrim dragonborn"
    ]

def getMediumPrompts():
    return {
        "painting": [
            "renaissance-style portrait",
            "Highly Detailed Digital Painting",
            "oil painting on canvas",
            "thick paint",
            "rembrandt van rijn",
            "golden age",
            "johannes vermeer",
            "frans hals",
        ],
        "fantasy": [
            "sci-fi",
            "dark",
            "dystopian",
            "futuristic",
            "neon lighting",
            "greg rutkowski",
            "zabrocki",
            "karlkka",
            "jayison devadas",
            "trending on artstation",
            "8k",
            "ultra wide angle",
        ],
        "comic": [
            "comic book cover",
            "retro comic",
            "illustration",
            "handdrawn",
            "vintage",
            "symmetrical"
            "vibrant",
            "poster by ilya kuvshinov katsuhiro",
            "magali villeneuve",
            "artgerm",
            "jeremy lipking",
            "michael garmash",
            "rob rey",
            "kentaro miura style",
            "trending on art station"
        ],
        "portrait": [
            "portrait",
            "Street photography photo",
            "photography",
            "dslr",
            "sharp focus",
            "high shutter speed",
            "depth of field",
            "cinematic",
            "low angle photograph",
            "upper body framing",
            "cgsociety",
            "photorealistic",
            "high resolution", # !!
            "highly detailed", # !!
            "intricate design", # !!
            "8k",
            "ray tracing",
            "filmed on Sony A7iii, 50mm, f/2.8",
            "film director James Cameron"
        ],
        "icon": [
            "appicon style",
            "mobile app logo design",
            "squared with round edges",
            "flat vector app icon",
            "vector art",
            "digital art"
            "centered image",
            "icon",
            "logo",
            "line art",
            "illustration",
            "minimalistic"
        ],
        "magazine": [
            "Cover of an award-winning magazine",
            "photo for magazine",
            "magazine cover",
            "craig mullins style",
        ]
    }

def main():
    mediumPrompts = getMediumPrompts()
    for char in getCharacters():
        for medium, prompts in mediumPrompts.items():
            img({
                "prompt": f"{char},{','.join(prompts)}",
                "batch_size":10,
                "seed": 152,
                "count": 1,
                "img_w": 512,
                "img_h": 512,
                "num_inference_steps": 10,
                "outputFile": f"{char}-{medium}",
            })

if __name__ == "__main__":
    main()