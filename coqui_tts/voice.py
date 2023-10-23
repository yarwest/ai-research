import argparse
import logging
import sys
import os
import uuid
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

    #logging.info('#### Available Coqui TTS models ####')
    #print(TTS().models)

    processID = uuid.uuid4()

    print("#### Process ID:", processID)

    logging.info('#### Initialising Coqui TTS ####')
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    languages = [ 'en', 'nl', 'multilingual' ]
    gender = [ 'female', 'male', 'mixed' ]
    datasets = {
        'en': {
            "female": {
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
                'jenny': [ 'jenny' ],
                'ek1': [ 'tacotron2' ],
            },
            "male": { },
            "mixed": {
                'vctk': [ 'vits', 'fast_pitch' ],
                'multi-dataset': [ 'tortoise-v2' ],
                'sam': ['tacotron-DDC'],
                'blizzard2013': ['capacitron-t2-c50','capacitron-t2-c150_v2'],
            },
        },
        'multilingual': { 'female': {}, 'male': {},'mixed':{'multi-dataset': ['xtts_v1','your_tts','bark'] }},
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

    succesful = []
    failed = []

    logging.info('#### Cycling available models and datasets ####')
    for lang in ['en','multilingual']:
        for gender in ['male','mixed']:
            for dataset in datasets[lang][gender]:
                for modelName in datasets[lang][gender][dataset]:
                    logging.info(f"#### Generating {lang} model {modelName} on dataset {dataset} ####")
                    try:
                        # init TTS model
                        model = TTS(model_name=f"tts_models/{lang}/{dataset}/{modelName}", progress_bar=False).to(device)
                        filePath = f"./out/{processID.hex}-{modelName}-{dataset}-{args.voice}.wav"
                        # Text to speech to a file
                        if lang == 'multilingual':
                            # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
                            model.tts_to_file(text=args.prompt, speaker_wav=voiceFilePath, language="en", file_path=filePath)
                        else:
                            model.tts_to_file(text=args.prompt, speaker_wav=voiceFilePath, file_path=filePath)
                        succesful.append(f"{modelName}-${dataset}")
                    except Exception as e:
                        failed.append(f"{modelName}-${dataset}")
                        logging.error(f"#### Failed to perform TTS for model {modelName} on dataset {dataset} ####", e)

    logging.info("#### Succesful ####", succesful)
    logging.info("#### Failed ####", failed)

if __name__ == "__main__":
    main(parse_args())