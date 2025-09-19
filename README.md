# Video emotion classifier

Serkan Sulun 2024

This program classifies the emotion of any video. The videos do not have to be from a specific class such as facial videos or cinematic trailers.

Install required Python libraries using: 

`pip install -r files/requirements.txt`

You can classify your own videos by providing the path as the following. Inside the folder `sample_data`, there are sample videos from the Ekman-6 dataset.

`python video_emotion_classifier.py --video_path sample_data/fear.mp4`

You can also provide YouTube links:

`python video_emotion_classifier.py --youtube_link "https://www.youtube.com/watch?v=3YaRvbQSjrk"`

The following additional results are also shown: Automatic Speech Recognition (ASR), audio event classification (BEATs), image captioning (CLIP), face detection and emotion classification, Optical Character Recognition (OCR).

The raw videos and the pretrained features for the Ekman-6 dataset is available on [Zenodo](https://zenodo.org/records/17159328)

Also see our [Trailer Genre Classifier](https://github.com/serkansulun/trailer-genre-classification).

Our paper is [available](https://arxiv.org/pdf/2410.21303).

If you use this work in your research, please consider citing our work.

```
@article{trailer,
  title = {Movie Trailer Genre Classification Using Multimodal Pretrained Features},
  author = {Sulun, Serkan and Viana, Paula and Davies, Matthew E.P.},
  year = {2024},
  journal = {Expert Systems with Applications},
  volume = {258},
  pages = {125209},
  issn = {0957-4174},
  doi = {10.1016/j.eswa.2024.125209},
}
```

```
@inproceedings{vemoclap,
  title = {VEMOCLAP: A Video Emotion Classification Web Application},
  shorttitle = {VEMOCLAP},
  author = {Sulun, Serkan and Viana, Paula and Davies, Matthew E. P.},
  year = {2024},
  month = oct,
  number = {arXiv:2410.21303},
  eprint = {2410.21303},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2410.21303},
  urldate = {2024-12-18},
  booktitle = {International Symposium on Multimedia},
}
```
