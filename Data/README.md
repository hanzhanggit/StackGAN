**Data**

1. Download our preprocessed char-CNN-RNN text embeddings for [birds](https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE) and [flowers](https://drive.google.com/open?id=0B3y_msrWZaXLaUc0UXpmcnhaVmM) and save them to `Data/`.
  - [Optional] Follow the instructions [here](https://github.com/reedscot/icml2016) to download the pretrained char-CNN-RNN text encoders and extract your own text embeddings.
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) image data. Extract them to `Data/birds/` and `Data/flowers/`, respectively.
3. Preprocess images.
  - For birds: `python ./misc/preprocess_birds.py`
  - For flowers: `python ./misc/preprocess_flowers.py`


**Skip-thought Vocabulary**
- [Download](https://github.com/ryankiros/skip-thoughts) vocabulary for skip-thought vectors to `Data/`.
