"""This script is intended to be used with https://github.com/kkoutini/passt_hear21"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from hear21passt.base30sec import load_model, get_scene_embeddings, get_timestamp_embeddings


def process_audios(model, input_dir: Path, output_dir: Path, audio_file_extension: str):
    for audio_fn in tqdm(input_dir.rglob("*" + audio_file_extension)):
        sub_path = audio_fn.relative_to(input_dir)
        # print(audio_fn)
        audio, sample_rate = torchaudio.load(input_dir / audio_fn)
        audio = torchaudio.functional.resample(audio, sample_rate, 32000)
        embed = get_scene_embeddings(audio, model).squeeze(0)
        embed_np = embed.cpu().numpy()
        (output_dir / sub_path).parent.mkdir(exist_ok=True)
        np.save((output_dir / sub_path).with_suffix(".npy"), embed_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract embeddings from audio files.')
    parser.add_argument('input_dir', help='directory containing audio files', type=Path)
    parser.add_argument('output_dir', help='target directory to write the embedding files to', type=Path)
    parser.add_argument('--ext', help='file extension of the audio files', default=".wav", type=str)

    args = parser.parse_args()
    model = load_model().cuda()
    process_audios(model, args.input_dir, args.output_dir, args.ext)
