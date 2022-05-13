## Simple Antifraud
****
This is a simplified voice antifraud system created as part of bachelor's thesis at [Moscow Polytechnic University](https://mospolytech.ru/). The system is based on a pre-trained DeepSpeech model, Naive Bayes classifier and TF-IDF vectorizer.

Project was done to illustrate the impact of performing adversarial attacks on this type of systems so it should not be used in production. Even if you think that DeepSpeech is protected enough, the classifier is vulnerable to the [Bayesian poisoning](https://en.wikipedia.org/wiki/Bayesian_poisoning) itself. 

This is some kind of [Damn-Vulnerable Service](https://github.com/vavkamil/awesome-vulnerable-apps) so you can get a flag if you will properly abuse it.

### Project structure

- `checkpoints` contains .ckpt files of pretrained DeepSpeech models. Pretrained models can be found [here](releases).
- `training` includes notebook with data preparation and fitting for NB Classifier and vectorizer.
- `pickles` folder are used to store them.
- `example.ipynb` can be used as a quick-start guide.

### Installation

Install deepspeech.pytorch:

```
git clone https://github.com/SeanNaren/deepspeech.pytorch
cd deepspeech.pytorch
pip install -r requirements.txt
pip install -e .
```

Clone this repository and run within it to install remaining dependencies:
```
pip install -r requirements.txt
```

### Mitigations

The robustness of original [LibriSpeech model]() can be increased using adversarial retraining with gaussian data augmentation. The example model can be found in [Releases](releases). You can also try to use another controls, described [here](https://www.enisa.europa.eu/publications/securing-machine-learning-algorithms).

To retrain a model with a new data [original trainig script](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/training.py) can be used. Simply replace 
```
model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )
```
with 

```
    model = DeepSpeech.load_from_checkpoint(
        cfg.checkpoint.filepath,
        freeze=True,
        learning_rate=0.0001
    )
```
so you can retrain it like `python3 train.py checkpoint.filepath=/path/to/file.ckpt`.
