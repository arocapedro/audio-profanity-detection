
import os
import librosa
import warnings
import argparse
from tqdm import tqdm
import soundfile as sf
import tensorflow as tf
from pathlib import Path

def main(original_dataset, output_dataset, target_format, SR):
    warnings.filterwarnings('ignore')
    all_files_dataset = tf.data.Dataset.list_files(str(f'{original_dataset}*/*/*'), shuffle=True)
    sample_rate = int(SR)
    for file_path in tqdm(all_files_dataset):
        decoded_path = tf.get_static_value(file_path).decode('utf-8')
        new_file_path = decoded_path.replace(original_dataset, output_dataset)
        # p = Path(new_file_path)
        # p.rename(p.with_suffix(f'.{target_format}'))
        pre, _ = os.path.splitext(new_file_path)
        new_file_path = pre + f".{target_format.lower()}"

        if os.path.exists(str(new_file_path)):
            continue

        Path( os.sep.join(new_file_path.split(os.sep)[0:-1])).mkdir(parents=True, exist_ok=True)

        y, _ = librosa.load(decoded_path)
        sf.write(file=str(new_file_path), data=y, samplerate=sample_rate, format=target_format.upper())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original_dataset')
    parser.add_argument('output_dataset')
    parser.add_argument('target_format')
    parser.add_argument('SR')
    args = parser.parse_args()

    main(**vars(args))