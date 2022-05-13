import torchaudio
import pickle
import numpy as np
from textblob import TextBlob
from data_maker import NoisePreprocessor
import art.estimators.speech_recognition as asr
from deepspeech_pytorch.model import DeepSpeech

class ModelBuilder:
    def __init__(
        self, 
        pretrained_art = 'librispeech',
        classifier_path = 'pickles/classifier.pkl',
        vectorizer_path = 'pickles/vectorizer.pkl',
        ):
        self.pretrained = pretrained_art
        try:
            with open(classifier_path, 'rb') as fid:
                self.clfr = pickle.load(fid)
            with open(vectorizer_path, 'rb') as fid:
                self.vc = pickle.load(fid)
        except:
            print('Pickled objects corrupted')
            exit(1)
    
    def _load_ckpt(self, checkpoint):
        model = DeepSpeech.load_from_checkpoint(checkpoint)
        return asr.PyTorchDeepSpeech(model=model)

    def _load_art(self):
        return asr.PyTorchDeepSpeech(pretrained_model=self.pretrained)

    def _set_helpers(self, loaded):
        setattr(loaded, 'vc', self.vc)
        setattr(loaded, 'clfr', self.clfr)
        return loaded
    
    def get_regular(self):
        regular = self._load_art()
        setattr(regular, 'type', 'regular')
        return self._set_helpers(regular)
    
    def get_retrain(self, type, checkpoint):
        retrain = self._load_ckpt(checkpoint)
        setattr(retrain, 'type', type)
        return self._set_helpers(retrain)


class SimpleAntifraudPart:
    def __init__(
        self, 
        model, 
        verbose = False,
        preprocessor = False,
        ):
        self.model = model
        self.verbose = verbose
        if preprocessor:
            noise = NoisePreprocessor.MED_NOISE
            self.preprocessor = NoisePreprocessor(noise)
            setattr(self.model, 'type', 'gauss')
        else:
            self.preprocessor = None

    def _get_predictor(self, audio):
        if self.preprocessor:
            return self._predict(
                self.preprocessor.apply_noise(audio)
            )
        return self._predict(audio)    

    def get_type(self):
        return self.model.type

    def _predict(self, audio):
        return self.model.predict(audio)
    
    def predict(self, filepath):
        audio = torchaudio.load(filepath)[0].numpy()
        return self._get_predictor(audio)

    def _transcribe(self, filepath):
        return self.predict(filepath)[0].lower()\
                                        .split()\
                                        
    def _correct(self, text):
        return [
                str(TextBlob(word).correct()) \
                for word in text
        ]

    def check_audio(self, filepath) -> bool:
        input_text = self._transcribe(filepath)
        if input_text == 'ADVERSARIAL':
            return False, 'You get a flag'.split()
        input_corrected = self._correct(input_text)
        
        pipe = sum(
            self.model.clfr.predict(
                self.model.vc.transform(
                    input_corrected
            )))

        ans = pipe > 0
        return ans if not self.verbose else ans, input_corrected

class SimpleAntifraud():
    def __init__(self, parts):
        self.parts = dict()
        for part in parts:
            self.parts[part.get_type()] = part
    
    def _check_audio(self, filepath, part_type) -> bool:
        return self.parts[part_type].check_audio(filepath)

    def _check_partly(self, filepath, type):
        if type in self.parts.keys():
            return self._check_audio(filepath, type)
        else:
            raise NameError('Model not initializated')

    def check_gauss(self, filepath):
        return self._check_partly(filepath, 'gauss')
    
    def check_gauss_retrain(self, filepath):
        return self._check_partly(filepath, 'gauss_retrain')

    def check_adv_retrain(self, filepath):
        return self._check_partly(filepath, 'adv_retrain')

    def check_regular(self, filepath):
        return self._check_partly(filepath, 'regular')

    def check_all(self, filepath):
        types = self.parts.keys()
        return zip(types, 
            [self._check_audio(filepath, t) for t in types]
            )
