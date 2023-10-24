import argparse
import logging
import sys
import json
import os
import whisper
import uuid
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
    parser.add_argument("--path", help="Path to audio file to be transcribed", type=str, required=False)

    return parser.parse_args()

def getFilePath(fileName):
    return os.path.join(os.path.dirname(__file__), fileName)

def new_dict(old_dict):
    n = old_dict.copy()
    [n.pop(key) for key in ['seek','tokens','temperature','avg_logprob','no_speech_prob','compression_ratio']]
    return n

def main(args):
    # Create output directory if not exists
    if not os.path.exists('./out'):
        os.makedirs('out/')

    processID = uuid.uuid4()

    print("---- Process ID:", processID)
    logging.info('---- Initialising Whisper AI ----')
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transcribe audio file
    AUDIO_FILE = os.path.join(os.path.dirname(__file__), f"yarno_voice.wav")

    try:
        # use the audio file as the audio source
        model = whisper.load_model("base")
        logging.info('---- Transcribing audio ----')
        for file in os.listdir(getFilePath("data/")):
            filePath = getFilePath(f"data/{file}")
            if not os.path.isfile(filePath):
                continue
            # TODO: fix (doesnt work rn)
            if file in os.listdir(getFilePath("out/")):
                logging.info(f"Already transcribed: {file}")
            result = model.transcribe(filePath, fp16=False, language="en")
            logging.info("Transcription: ", result)
            print(result)
            for key in result:
                print(f"{key}")
            segments = map(new_dict,result["segments"])
            with open(f'./out/{processID}-{file}.txt', 'a+') as outfile:
                outfile.write(json.dumps(list(segments), indent=4))
    except Exception as e:
        logging.info(f"---- Failed to transcribe ----", e)

if __name__ == "__main__":
    main(parse_args())