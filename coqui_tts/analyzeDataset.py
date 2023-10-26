import argparse
import logging
import sys
import os
import uuid
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.notebook import tqdm
from multiprocessing import Pool
from scipy.stats import norm
from matplotlib import pylab as plt
import librosa
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

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
    parser.add_argument("--dataset_dir", help="Directory containing dataset to be loaded", type=str, required=False)

    return parser.parse_args()

def getFilePath(fileName):
    return os.path.join(os.path.dirname(__file__), fileName)

# custom formatter implementation
def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "Yarno Boelens"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items

def validateAudioFiles(files):
    logging.info(" > Number of audio files: {}".format(len(files)))
    print(files[1])

    # check wavs if exist
    wav_files = []
    for file in files:
        wav_file = file["audio_file"].strip()
        wav_files.append(wav_file)
        if not os.path.exists(wav_file):
            logging.error(f"Couldn't find: {wav_file}")

    # show duplicate files
    c = Counter(wav_files)
    logging.info(f"Duplicate files: {[file for file, count in c.items() if count > 1]}")

    return wav_files

def load_item(item):
    text = item["text"].strip()
    file_name = item["audio_file"].strip()
    audio, sr = librosa.load(file_name, sr=None)
    audio_len = len(audio) / sr
    text_len = len(text)
    return file_name, text, text_len, audio, audio_len

def countWords(data):
    # count words in the dataset
    w_count = Counter()
    for item in tqdm(data):
        text = item[1].lower().strip()
        for word in text.split():
            w_count[word] += 1

    logging.info(" > Number of words: {}".format(len(w_count)))
    return w_count

def audioLenPerCh(data):
    logging.info("Avg audio length per char")
    for item in data:
        if item[-1] < 2:
            logging.info(item)

    sec_per_chars = []
    for item in data:
        text = item[1]
        dur = item[-1]
        sec_per_char = dur / len(text)
        sec_per_chars.append(sec_per_char)
    # sec_per_char /= len(data)
    # logging.info(sec_per_char)

    mean = np.mean(sec_per_chars)
    std = np.std(sec_per_chars)
    logging.info(mean)
    logging.info(std)

    dist = norm(mean, std)

    # find irregular instances long or short voice durations
    for item in data:
        text = item[1]
        dur = item[-1]
        sec_per_char = dur / len(text)
        pdf = norm.pdf(sec_per_char)
        if pdf < 0.39:
            logging.error(f"Irregular length audio found: {item}")

def textVsAudioLength(data):
    text_vs_durs = {}  # text length vs audio duration
    text_len_counter = Counter()  # number of sentences with the keyed length
    for item in tqdm(data):
        text = item[1].lower().strip()
        text_len = len(text)
        text_len_counter[text_len] += 1
        audio_len = item[-1]
        try:
            text_vs_durs[text_len] += [audio_len]
        except:
            text_vs_durs[text_len] = [audio_len]

    # text_len vs avg_audio_len, median_audio_len, std_audio_len
    text_vs_avg = {}
    text_vs_median = {}
    text_vs_std = {}
    for key, durs in text_vs_durs.items():
        text_vs_avg[key] = np.mean(durs)
        text_vs_median[key] = np.median(durs)
        text_vs_std[key] = np.std(durs)

    plotData(text_vs_avg, text_vs_median, text_vs_std, text_len_counter)

def plotData(text_vs_avg, text_vs_median, text_vs_std, text_len_counter):
    # Plot Dataset Statistics
    plt.title("text length vs mean audio duration")
    plt.scatter(list(text_vs_avg.keys()), list(text_vs_avg.values()))
    plt.show()
    plt.title("text length vs median audio duration")
    plt.scatter(list(text_vs_median.keys()), list(text_vs_median.values()))
    plt.show()
    plt.title("text length vs STD")
    plt.scatter(list(text_vs_std.keys()), list(text_vs_std.values()))
    plt.show()
    plt.title("text length vs # instances")
    plt.scatter(list(text_len_counter.keys()), list(text_len_counter.values()))
    plt.show()

def plotWordCount(w_count):
    # Check words frequencies
    w_count_df = pd.DataFrame.from_dict(w_count, orient='index')
    w_count_df.sort_values(0, ascending=False, inplace=True)
    w_count_df
    # check a certain word
    print(f"Count of 'create' {w_count_df.at['create', 0]}")
    # fequency bar plot - it takes time!!
    w_count_df.plot.bar()


def main(args):
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset_dir)
    dataset_config = BaseDatasetConfig(formatter="ljspeech", meta_file_train="metadata.txt", language="en-us", path=dataset_path)

    processID = uuid.uuid4()

    print("---- Process ID:", processID)
    logging.info('---- Loading dataset ----')

    # transcribe audio file
    AUDIO_FILE = os.path.join(os.path.dirname(__file__), f"yarno_voice.wav")
    NUM_PROC = 8
    try:
        # load training samples
        train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)
        if eval_samples is not None:
            items = train_samples + eval_samples
        else:
            items = train_samples

        validateAudioFiles(items)
        
        # This will take a while depending on size of dataset
        if NUM_PROC == 1:
            data = []
            for m in tqdm(items):
                data += [load_item(m)]
        else:
            with Pool(8) as p:
                data = list(tqdm(p.imap(load_item, items), total=len(items)))

        w_count = countWords(data)
        
        audioLenPerCh(data)

        textVsAudioLength(data)

        plotWordCount(w_count)
        
  
    except Exception as e:
        logging.info(f"---- Failed to load dataset ----", e)

if __name__ == "__main__":
    main(parse_args())