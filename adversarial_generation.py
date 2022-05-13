import time
from os import path
from typing import Callable, List
import torch
import torchaudio
import numpy as np
import art.estimators.speech_recognition as asr
from art.attacks.evasion import ImperceptibleASRPyTorch

import warnings
warnings.filterwarnings(action='ignore')

def save_to_txt(dest_dir, filename, content):
    with open(path.join(dest_dir, filename), 'a') as f:
        f.write(content)

def time_to_file(
    file_maker : Callable,
    dest_dir : str = 'timing',
):
    def decorator(function):
        def wrapper(model, filepath, transcription):
            start_time = time.time()
            result = function(model, filepath, transcription)
            elapsed = time.time() - start_time
            sample = path.split(filepath)[-1]
            file_maker(
                dest_dir, 
                sample.split('.')[0],
                f'Time elapsed: {elapsed}s for {sample}\n'
            )
            return result
        return wrapper
    return decorator

@time_to_file(save_to_txt)
def make_adversarial(model, filepath, labels):
    adversarial = ImperceptibleASRPyTorch(model)
    audio = load_np_audio(filepath)
    waveform_np = adversarial.generate(audio, labels)
    return waveform_np

def create_args(sample_list, transcription):
    transc = list()
    for sample in sample_list:
        rec = sample.split('_')[0]
        if rec in transcription.keys():
            transc.append(transcription[rec])
    return zip(sample_list, transc)

def load_np_audio(filepath):
    return torchaudio.load(filepath)[0].numpy()

def save_np_audio(array, filename, destdir):
    tensor = torch.from_numpy(array)
    filepath = path.join(destdir, filename)
    torchaudio.save(filepath, tensor, 16000)

def create_advs(
    model : asr.PyTorchDeepSpeech,
    source_dir : str, 
    dest_dir : str, 
    samples : List[str], 
    transcriptions : List[str]
) -> List[np.ndarray]:
    advs = []
    for args in create_args(samples, transcriptions):
        advs.append(make_adversarial(model, 
                         path.join(source_dir, args[0]), 
                         np.array([args[1]])
        ))
        save_np_audio(advs[-1], args[0], dest_dir)
    return advs
