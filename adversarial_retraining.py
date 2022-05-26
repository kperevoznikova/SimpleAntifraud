import os
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from itertools import product
from deepspeech_pytorch.model import DeepSpeech
import art.estimators.speech_recognition as asr
from adversarial_generation import create_advs
from data_maker import DataMaker

def ckpt_to_asr(ckptpath):
    model = DeepSpeech.load_from_checkpoint(ckptpath)
    model.eval() # go back from freeze
    return asr.PyTorchDeepSpeech(model=model)

def make_rrs(mod):
    l = [0,1]
    for i in range(1,mod):
        if mod%i: l.append(i)
    return l

def shuffle_dict(transc):
    '''Cyclically changes samples and transcriptions using reduced residue system'''
    decart = list(product(transc.keys(), transc.values()))
    dicts = list()
    for i,j in enumerate(make_rrs(len(decart))):
        el = decart[(j*7 + 3) % 25]
        if i%5 == 0:
            dicts.append({})
        dicts[i//5][el[0]] = el[1]
    return dicts

def make_transc(shuffle=False):
    config = Path('adversarial_transcriptions.yml').read_text()
    transcriptions = yaml.load(config)
    if not shuffle:
        return transcriptions
    return shuffle_dict(transcriptions)

class Retrainer:
    def __init__(
        self,
        logger,
        checkpoint,
        from_dir,
        tmp_dir
        ):
        self.logger = logger
        self.checkpoint = checkpoint
        self.set_from_dir(from_dir)
        self.tmp_dir = tmp_dir
        self.aug_dir = 'transcription_prepared/augmented'
        self.vals_dir = 'transcription_prepared/vals'
        self.subtmp = -1

    @staticmethod
    def select_all_samples(samples_dir):
        names = list()
        for i in Path(samples_dir).iterdir():
            if i.is_file():
                names.append(str(i.name))
        return names

    @staticmethod
    def _read_config(file):
        config = Path(file).read_text()
        return yaml.load(config)
    
    @staticmethod
    def n_files(path):
        return len(list(path.iterdir()))

    @staticmethod
    def find_last_ckpt(outputs, in_logs=False):
        outputs = Path(outputs)
        dates = list(outputs.iterdir())
        for i in sorted(dates, reverse=True):
            times = list(i.iterdir())
            for j in sorted(times, reverse=True):
                if not in_logs:
                    ckpt_folder = j / 'checkpoints'
                else:
                    logs = 'lightning_logs/version_0/checkpoints'
                    ckpt_folder = j / logs
                if ckpt_folder.exists():
                    return str(ckpt_folder / 'last.ckpt')
                else:
                    break
        if not in_logs:
            return Retrainer.find_last_ckpt(outputs, True)
        else:
            raise FileNotFoundError

    def set_from_dir(self, from_dir):
        self.from_dir = from_dir
        self.samples = self.select_all_samples(from_dir)

    def set_checkpoint(self, checkpoint):
        logger.info(f'New checkpoint is {checkpoint}')
        i = self.n_files(Path('checkpoints'))
        newpath = f'checkpoints/noised_{i}.ckpt'
        shutil.copy(checkpoint, newpath)
        self.checkpoint = newpath

    def make_subtmp(self, i):
        name = f'{self.tmp_dir}_{i}'
        self.subtmp += 1
        Path(name).mkdir(exist_ok=True)
        return name

    def create_batch_advs(self):
        for i, t in enumerate(make_transc(shuffle=True)):
            self.logger.debug(f'Current transcriptions is:\n{t}')
            create_advs(
                ckpt_to_asr(self.checkpoint), 
                self.from_dir, 
                self.make_subtmp(i), 
                self.samples, 
                t
            )
    
    def concat_batches(self):
        tmp = Path(self.tmp_dir)
        tmp.mkdir(exist_ok=True)

        for i in range(self.subtmp, -1, -1):
            subdir = f'{self.tmp_dir}_{i}'
            for sample in Path(subdir).iterdir():
                name = sample.name.split('.')
                shutil.copy(sample, tmp / f'{name[0]}_{i}.wav')
            shutil.rmtree(subdir)
            self.subtmp -= 1

    def _make_structure(self):
        txt = Path(self.tmp_dir) / 'txt'
        txt.mkdir(exist_ok=True)
        wav = Path(self.tmp_dir) / 'wav'
        wav.mkdir(exist_ok=True)
        return txt, wav
    
    def _fit_structure(self, wav):
        for sample in Path(self.tmp_dir).iterdir():
            if sample.is_file():
                shutil.move(sample, wav / sample.name)

    def _make_transcriptions(self, wav, txt):
        config = 'true_transcriptions.yml'
        transcriptions = self._read_config(config)

        for sample in wav.iterdir():
            type = sample.name.split('_')[0]
            name = sample.name.split('.')[0]
            transc = Path(txt / f'{name}.txt')
            transc.write_text(transcriptions[type])

    def _calc_val_size(self):
        samples_dir = Path(self.aug_dir) / 'txt'
        size = self.n_files(samples_dir)
        return int(size * .25)

    def make_train_sets(self):
        txt, wav = self._make_structure()
        self._fit_structure(wav)
        self._make_transcriptions(wav, txt)

        Path('manifests').mkdir(exist_ok=True)
        dm = DataMaker(
            samples_folder=self.tmp_dir,
            dest_path=self.aug_dir,
        )
        dm.apply_noise()
        dm = DataMaker(
            samples_folder=self.tmp_dir,
            dest_path=self.vals_dir,
        )
        dm.create_vals(self._calc_val_size())

    def _retrain(self):
        # TODO: check if retrain script is changed
        os.system('''
        script=$HOME/deepspeech.pytorch/train.py

        train_manifest='manifests/train_noise.json'
        val_manifest='manifests/val_noise.json'

        python3 $script \
        checkpoint.save_last=true \
        checkpoint.monitor=wer \
        checkpoint.save_top_k=1 \
        checkpoint.verbose=true \
        checkpoint.filepath=`pwd`/{} \
        data.train_path=$train_manifest \
        data.val_path=$val_manifest \
        data.num_workers=8 \
        data.batch_size=8 \
        trainer.gpus=1 \
        trainer.accelerator=gpu \
        trainer.max_steps=-1 \
        trainer.max_epochs=10 \
        trainer.strategy=ddp \
        trainer.gradient_clip_val=400 \
        '''.format(self.checkpoint))

    def prepare(self):
        self.logger.info('Making new samples')
        self.create_batch_advs()
        self.concat_batches()
        self.make_train_sets()
    
    def clear(self):
        self.logger.info('Clearing samples directory')
        shutil.rmtree(self.tmp_dir)
        shutil.rmtree(self.aug_dir)
        shutil.rmtree(self.vals_dir)
        shutil.rmtree('manifests')

    def retrain(self):
        self.prepare()
        self._retrain()
        self.logger.info('Retraining complete')
        trained = self.find_last_ckpt('outputs')
        self.set_checkpoint(trained)
        self.clear()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Perform Adversarial Retraining with augmentation'
        )
    parser.add_argument(
        '--checkpoint', 
        default='checkpoints/librispeech_pretrained_v3.ckpt',
        help='Lightning model checkpoint to retrain'
        )
    parser.add_argument(
        '--audio', 
        default='transcription_prepared/fraud', 
        help='Folder with audio which will be turned into adversarial'
        )
    parser.add_argument(
        '--tmp', 
        default='transcription_prepared/tmpfolder',
        help='Folder to store generated data'
        )
    parser.add_argument(
        '--limit', 
        default=3,
        help='Amount of iterations to perform'
        )
    args = parser.parse_args()

    logger = logging.getLogger('retraining')
    logger.setLevel(logging.DEBUG)

    retrainer = Retrainer(
        logger=logger,
        checkpoint=args.checkpoint,
        from_dir=args.audio,
        tmp_dir=args.tmp
    )

    for i in range(args.limit):
        retrainer.retrain()
