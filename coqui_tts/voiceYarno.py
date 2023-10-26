import argparse
import logging
import sys
import os
import uuid
import torch
from TTS.api import TTS
# VitsConfig: all model related values for training, validating and testing.
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.config import load_config
# from TTS.utils.synthesizer import Synthesizer

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
    parser.add_argument("--prompt", help="Prompt for audio generation", type=str, required=True)

    return parser.parse_args()

def modelPath(file):
    modelPath = os.path.join(os.path.dirname(__file__), f"YourTTS-EN-yarnoData-October-26-2023_04+41PM-e09f1bc/")
    return os.path.join(modelPath, file)

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('out/')

    speakerPth = modelPath('speakers.pth')
    configPath = modelPath('config.json')

    processID = uuid.uuid4()

    print("#### Process ID:", processID)

    # Get device
    cuda = torch.cuda.is_available()
 
    try:
        logging.info('#### Initialising Vits Config ####')
        config = VitsConfig()
        model = Vits(config)
        
        # init TTS model
        logging.info(f"#### Initializing Yarno model ####")
        config = load_config(configPath)
        model = Vits.init_from_config(config)
        print(model)
        model.load_checkpoint(config=config, checkpoint_path=speakerPth, eval=True)

        logging.info(f"#### Synthesizing audio ####")
        filePath = f"./out/yarno-{processID.hex}.wav"
        model.tts_to_file(text=args.prompt, speaker_wav=filePath, file_path=filePath)
        
        # Text to speech to a file
        # synthesizer = Synthesizer(tts_speakers_file=speakerPth,
        #                           tts_config_path=configPath,
        #                           use_cuda=cuda)
        # wav = synthesizer.tts(text=args.prompt)
        # logging.info(f"#### Saving audio ####")
        # synthesizer.save_wav(wav, filePath)
    except Exception as e:
        logging.info(f"#### Failed to perform TTS ####", e)

if __name__ == "__main__":
    main(parse_args())