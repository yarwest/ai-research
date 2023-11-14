import argparse
import logging
import sys
import json
import os
import datetime
import whisper
import uuid
import torch
import contextlib
import wave
from pydub import AudioSegment
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device(device))

from pyannote.audio import Audio
from pyannote.core import Segment

from sklearn.cluster import AgglomerativeClustering

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
    parser.add_argument("--speakers", help="Number of unique speakers in audio file", type=int, default=1, required=False)
    parser.add_argument("--split", help="Pass True to split original audio file into seperate files for each detected speech segment", type=bool, required=False)

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

def toWav(path, fullPath):
    logging.info('---- Converting file to single channel WAV ----')
    file = path.split('.')[0]
    outPath = f'{getFilePath(f"{DATA_DIR}/{file}-mono")}.wav'
    subprocess.call(['ffmpeg', '-i', fullPath, '-ac', '1', outPath, '-y'])
    return outPath

def segment_embedding(segment, duration, path):
  audio = Audio()
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])

def time(secs):
  return datetime.timedelta(seconds=round(secs))

def plot(embeddings, segments, labels):
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot the clusters
    plt.figure(figsize=(10, 8))
    for i, segment in enumerate(segments):
        speaker_id = labels[i] + 1
        x, y = embeddings_2d[i]
        color='green'
        if(speaker_id > 1):
            color='red'
        plt.scatter(x, y,
                    edgecolor=color,
                    marker=f'$\\speaker{speaker_id}$',
                    label=f'SPEAKER {speaker_id}')

    plt.title("Speaker Diarization Clusters (PCA Visualization)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig('./out/speaker-distribution.png')

def processFile(file, args):
    originalFilePath = getFilePath(f"{DATA_DIR}/{file}")
    if not os.path.isfile(originalFilePath):
        return
    filePath = toWav(file, originalFilePath)
    model = whisper.load_model("base")
    logging.info('---- Transcribing audio ----')
    segments = transcribe(file, filePath, model)

    if(args.split):
        baseAudio = AudioSegment.from_wav(filePath)
        logging.info('---- Slicing audio ----')
        exportedSegments = sliceSegments(segments, baseAudio)

        # store segment data in metadata.txt
        with contextlib.closing(open(f"./{OUT_DIR}/{DATASET_DIR}/metadata.txt", 'a+')) as outfile:
            for segID in exportedSegments:
                outfile.write(f"{segID}|{exportedSegments[segID]}|{exportedSegments[segID]}\n")

    if(args.speakers > 1):
        logging.info('---- Speaker detection ----')
        
        with contextlib.closing(wave.open(filePath,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment, duration, filePath)

        embeddings = np.nan_to_num(embeddings)

        clustering = AgglomerativeClustering(args.speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        logging.info('---- Saving speaker detection transcript ----')
        f = open(f"./{OUT_DIR}/{file}-speakers.txt", "w", encoding="utf-8")

        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
            f.write(segment["text"][1:] + ' ')
        f.close()

        plot(embeddings, segments, labels)

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
        if(args.path):
            processFile(args.path, args)
        else:
            # Loop through data folder
            logging.info('---- Starting loop of data dir ----')
            for file in os.listdir(getFilePath(f"{DATA_DIR}/")):
                processFile(file, args)

    except Exception as e:
        logging.info(f"---- Failed to transcribe ----", e)

if __name__ == "__main__":
    main(parse_args())