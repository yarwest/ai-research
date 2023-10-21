import argparse
import logging
import sys
import os

import torch
from TTS.api import TTS

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
    parser.add_argument("--voice", default="indian_voice", help="Name of voice sample file (.wav in data folder) to be cloned for tts", type=str, required=False)
    
    return parser.parse_args()

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('out/')

    print(TTS().models)
    logging.info('=== available models ===')

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    languages = [ 'en', 'nl', 'multilingual' ]
    datasets = {
        'en': {
            'ljspeech': [
                'tacotron2-DDC',
                'tacotron2-DDC_ph',
                'tacotron2-DCA',
                'speedy-speech',
                'vits',
                'vits--neon',
                'glow-tts',
                'hifigan_v2',
                'fast_pitch',
                'overflow',
                'neural_hmm'
            ],
            'vctk': [ 'vits', 'fast_pitch' ],
            'sam': ['tacotron-DDC'],
            'blizzard2013': ['capacitron-t2-c50','capacitron-t2-c150_v2'],
            'ek1': [ 'tacotron2' ],
            'multi-dataset': [ 'tortoise-v2' ],
            'jenny': [ 'jenny' ] },
        'multilingual': { 'multi-dataset': ['xtts_v1','your_tts','bark'] },
        'uk': { 'mai': ['glow-tts', 'vits']},
        'nl': {'mai': ['tacotron2-DCA'],'css10':['vits']},
        'pl': {'mai_femaile': ['vits']}
    }
    models = {
        'en': [
            'tacotron2'
            'tacotron2-DDC',
            'tacotron2-DDC_ph',
            'tacotron2-DCA',
            'speedy-speech',
            'vits',
            'glow-tts',
            'hifigan_v2'
        ],
        'multilingual': [ 
            'xtts_v1',
        ]
    }

    voiceFilePath = os.path.join(os.path.dirname(__file__), f"data/{args.voice}.wav");

    for lang in ['en','multilingual']:
        for dataset in datasets[lang]:
            for modelName in datasets[lang][dataset]:
                logging.info(f"### Generating {lang} model {modelName} on dataset {dataset} ###")
                try:
                    # init TTS model
                    model = TTS(model_name=f"tts_models/{lang}/{dataset}/{modelName}", progress_bar=False).to(device)
                    # Text to speech to a file
                    if lang == 'multilingual':
                        # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
                        model.tts_to_file(text=args.prompt, speaker_wav=voiceFilePath, language="en", file_path=f"./out/{modelName}-{dataset}-{args.voice}.wav")
                    else:
                        model.tts_to_file(text=args.prompt, speaker_wav=voiceFilePath, file_path=f"./out/{modelName}-{dataset}-{args.voice}.wav")
                except Exception as e:
                    logging.error(f"### Failed to perform TTS for model {modelName} on dataset {dataset} ###", e)

if __name__ == "__main__":
    main(parse_args())