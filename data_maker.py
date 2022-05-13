import shutil
from pathlib import Path
import numpy as np
from deepspeech_pytorch.data.utils import create_manifest
from adversarial_generation import save_np_audio, load_np_audio

class NoisePreprocessor:

    HARD_NOISE = .1
    MED_NOISE = .01
    LIGHT_NOISE = .005

    def __init__(self, sigma=.01):
        self.sigma = sigma

    def set_noise(self, sigma):
        if sigma < 1:
            self.sigma = sigma
        else:
            raise ValueError

    def apply_noise(self, x):
        noise = np.random.normal(x, self.sigma, x.shape)
        return noise.astype(np.float32)

class DataMaker():

    LIBRISPEECH_MAX = 281241
    
    def __init__(
        self,
        samples_folder : str,
        dest_path : str,
        manifest_path : str = 'manifests',
        num_workers: int = 1
        ):
        self.num_workers = num_workers
        self.manifest_path = Path(manifest_path)
        self.dest_path = Path(dest_path)
        self.dest_wav = self.dest_path / 'wav'
        self.dest_txt = self.dest_path / 'txt'
        self.samples_wav = Path(samples_folder) / 'wav'
        self.samples_txt = Path(samples_folder) / 'txt'
        self.prep = NoisePreprocessor()
    
    def _create_manifest(self, prefix):
        output_name = f'{prefix}_noise.json'
        create_manifest(
                        str(self.dest_path), 
                        output_name,
                        self.manifest_path,
                        self.num_workers
                        )

    def _save_sample(self, name, text, audio):
        save_np_audio(audio, f'{name}.wav', str(self.dest_wav))
        (self.dest_txt / f'{name}.txt').write_text(text)
    
    def _get_text(self, name):
        return (self.samples_txt / f'{name}.txt').read_text()

    def _apply_noise(self, sample, times):
        audio = load_np_audio(sample)
        name = sample.name.rstrip('.wav')
        text = self._get_text(name)
        for t in range(times):
            audio_noised = self.prep.apply_noise(audio)
            self._save_sample(f'{name}_{t}', text, audio)

    def _make_dirs(self):
        self.dest_path.mkdir()
        self.dest_wav.mkdir()
        self.dest_txt.mkdir()

    def apply_noise(self, prefix='train', times=3):
        self._make_dirs()
        for sample in self.samples_wav.iterdir():
            self._apply_noise(sample, times)
        self._create_manifest(prefix)

    def _val_random(self, size):
        gen = np.random.default_rng()
        return sorted(gen.choice(
                self.LIBRISPEECH_MAX, 
                size=size, 
                replace=False
                ))

    def _copy_sample(self, wav):
        name = wav.name.rstrip('.wav')
        shutil.copy(str(wav), self.dest_wav)
        txt = self.samples_txt / f'{name}.txt'
        shutil.copy(str(txt), self.dest_txt)

    def _create_random(self, size):
        indexes = self._val_random(size)
        for i, wav in enumerate(self.samples_wav.iterdir()):
            if i in indexes:
                self._copy_sample(wav)

    def _create_iterate(self, size):
        for i, wav in enumerate(self.samples_wav.iterdir()):
            if i == size:
                break
            self._copy_sample(wav)

    def create_vals(self, size, random=False):
        self._make_dirs()
        if random:
            self._create_random(size)
        else:
            self._create_iterate(size)
        self._create_manifest(prefix='val')
