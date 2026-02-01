"""
Script to filter dataset based on word timestamps
"""
from argparse import ArgumentParser
import shutil
from tqdm import tqdm
from loguru import logger
import os
import json
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from generation.transcription.parakeet import Parakeet
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--subset_dir", type=str, default="data/QuadVoxBench/PublicFigure")
    parser.add_argument("-l", "--language", type=str, default="en")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    args = parser.parse_args()

    model = Parakeet()
    with open(os.path.join(args.subset_dir, "meta.json"), "r") as f:
        meta_data = json.load(f)
    new_meta_data = []
    for i in tqdm(range(0, len(meta_data), args.batch_size)):
        batch = meta_data[i:i+args.batch_size]
        audio_paths = [os.path.join(args.subset_dir, item["audio"]["real"]) for item in batch]
        try:
            texts = model.transcribe(audio_paths, args.language)
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            continue
        for j, item in enumerate(batch):
            item["text"] = texts[j][0]
            word_timestamps = texts[j][1]
            if len(word_timestamps) < 3:
                logger.warning(f"Word timestamps are less than 3: {word_timestamps}")
                logger.warning(f"Audio path: {audio_paths[j]}")
                if os.path.exists(audio_paths[j]):
                    shutil.move(audio_paths[j], f"{audio_paths[j]}.backup")
                if os.path.exists(audio_paths[j].replace("real", "fake")):
                    shutil.move(audio_paths[j].replace("real", "fake"), f"{audio_paths[j].replace('real', 'fake')}.backup")
            else:
                new_meta_data.append(item)
    with open(os.path.join(args.subset_dir, "meta_filtered.json"), "w") as f:
        json.dump(new_meta_data, f, indent=2, ensure_ascii=False)