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
        "jedi, star wars",
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
        "skyrim dragonborn",
        "popeye",
        "captain jack sparrow, pirate",
        "power rangers"
    ]

def getMediumPrompts():
    return {
        "lineart": [
            "highly detailed intricate line drawing",
            "clean simple line art",
            "black and white pencil drawing",
            "handdrawn cartoon on paper",
            "black and white coloring book page",
            "coloring book line art by artgerm and greg rutkowski and johanna basford and alphonse mucha",
        ],
        "painting": [
            "renaissance-style portrait",
            "Highly Detailed Digital Painting",
            "oil painting on canvas",
            "thick paint",
            "style of rembrandt van rijn",
            "golden age",
            "style of johannes vermeer",
            "style of frans hals",
            
            "highly detailed", # !!
        ],
        "fantasy": [
            "sci-fi",
            "dark",
            "dystopian",
            "futuristic",
            "neon lighting",
            "style of greg rutkowski",
            "style of zabrocki",
            "style of karlkka",
            "style of jayison devadas",
            "trending on artstation",

            "ultra wide angle",
            "photorealistic",
            "high resolution", # !!
            "highly detailed", # !!
            "intricate design", # !!
            "8k",
            "ray tracing",
        ],
        "comic": [
            "comic book cover",
            "retro comic",
            "illustration",
            "handdrawn",
            "pencil on paper",
            "vintage",
            "symmetrical"
            "vibrant",
            "style of  ilya kuvshinov katsuhiro",
            "style of magali villeneuve",
            "style of artgerm",
            "style of jeremy lipking",
            "style of michael garmash",
            "style of rob rey",
            "kentaro miura style",
            "trending on art station",

            "highly detailed", # !!
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
            "8k",
            "ray tracing",
            "filmed on Sony A7iii, 50mm, f/2.8",
            "film director James Cameron",
            
            "photorealistic",
            "high resolution", # !!
            "highly detailed", # !!
            "intricate design", # !!
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