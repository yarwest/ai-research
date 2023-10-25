import argparse
import logging
import sys
import json
import os
import whisper
import uuid
from pydub import AudioSegment

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

def initDirs():
    # Create output directory if not exists
    if not os.path.exists(f"./{OUT_DIR}"):
        os.makedirs(f"./{OUT_DIR}")
    if not os.path.exists(f"./{OUT_DIR}/{RAW_DIR}"):
        os.makedirs(f"./{OUT_DIR}/{RAW_DIR}")
    if not os.path.exists(f"./{OUT_DIR}/{DATASET_DIR}"):
        os.makedirs(f"./{OUT_DIR}/{DATASET_DIR}")
    if not os.path.exists(f"./{OUT_DIR}/{DATASET_DIR}/{WAVS_DIR}"):
        os.makedirs(f"./{OUT_DIR}/{DATASET_DIR}/{WAVS_DIR}")

def getFilePath(fileName):
    return os.path.join(os.path.dirname(__file__), fileName)

def format_segment(old_dict):
    n = old_dict.copy()
    for key in ['seek','tokens','temperature','avg_logprob','no_speech_prob','compression_ratio']:
        n.pop(key)
    return n

def isCompleteScentence(text):
    if text.split(" ").__len__() <= 2:
        return False
    return text.endswith(".") or text.endswith("?") or text.endswith("!")

def transcribe(file, filePath, model):
    if f"{file}.txt" in os.listdir(getFilePath(f"{OUT_DIR}/{RAW_DIR}")):
        logging.info(f"Already transcribed: {file}")
        with open(getFilePath(f"{OUT_DIR}/{RAW_DIR}/{file}.txt"), 'r') as file:
            return json.load(file)
    
    result = model.transcribe(filePath, fp16=False, language="en")
    segments = [format_segment(s) for s in result["segments"]]
    with open(f"./{OUT_DIR}/{RAW_DIR}/{file}.txt", 'a+') as outfile:
        outfile.write(json.dumps(segments, indent=4))
    return segments

def sliceSegments(segments, baseAudio):
    exportedSegments = {}
    previousSegments = []
    previousStart = 0

    # segment: { id: int, start: float, end: float, text: string }
    for segment in segments:
        text = segment['text'].strip()
        if isCompleteScentence(text):
            if previousSegments.__len__() > 0:
                t1 = previousStart * 1000 #Works in milliseconds
            else:
                t1 = segment['start'] * 1000 #Works in milliseconds
            
            t2 = segment['end'] * 1000
            newAudio = baseAudio[t1:t2]
            audioID = uuid.uuid4()
            newAudio.export(f"./{OUT_DIR}/{DATASET_DIR}/{WAVS_DIR}/{audioID}.wav", format="wav")
            previousSegments.append(text)
            exportedSegments[audioID] = " ".join(previousSegments)
            previousSegments = []
            previousStart = 0
        else:
            if previousSegments.__len__() == 0:
                previousStart = segment['start']
            previousSegments.append(text)

    return exportedSegments
    

DATA_DIR = 'data'
OUT_DIR = 'out'
RAW_DIR = 'raw'
DATASET_DIR = 'yarnoDataset'
WAVS_DIR = 'wavs'

def main(args):
    initDirs()
    processID = uuid.uuid4()
    print("---- Process ID:", processID)
    logging.info('---- Initialising Whisper AI ----')

    try:
        model = whisper.load_model("base")
        logging.info('---- Starting loop of data dir ----')
        # Loop through data filder
        for file in os.listdir(getFilePath(f"{DATA_DIR}/")):
            filePath = getFilePath(f"{DATA_DIR}/{file}")
            if not os.path.isfile(filePath):
                continue

            logging.info('---- Transcribing audio ----')
            segments = transcribe(file, filePath, model)
            
            baseAudio = AudioSegment.from_wav(filePath)
            
            logging.info('---- Slicing audio ----')
            exportedSegments = sliceSegments(segments, baseAudio)

            # store segment data in metadata.txt
            with open(f"./{OUT_DIR}/{DATASET_DIR}/metadata.txt", 'a+') as outfile:
                for segID in exportedSegments:
                    outfile.write(f"{segID}.wav|{exportedSegments[segID]}|{exportedSegments[segID]}\n")
    except Exception as e:
        logging.info(f"---- Failed to transcribe ----", e)

if __name__ == "__main__":
    main(parse_args())