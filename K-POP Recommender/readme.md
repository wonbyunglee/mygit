<K-POP Title Recommendation Based on Multilingual-BERT>

Introduction
- The application of deep learning technology is increasingly expanding to content generation such as image, text, and music.
- Increasing interest in the field of automatically learning music styles using different corpora and creating new music content based on them.
" Development of a model that predicts and recommends song titles by analyzing text data of song lyrics"

There is no model learned as Korean lyrics, there are many studies on lyrics generation, but there is a lack of research on title generation.
In addition, methods used in lyrics generation studies are mainly limited to RNN or GRU.
In this project, various model tests other than RNN and GRU are performed.

Dataset

2000-2023 Oct. Melon chart 100 dataset
From : https://github.com/EX3exp/Kpop-lyric-datasets
We extract data from the original data in json format from 2010 to October 2023.
It consists of 18,000 songs and 900,000 lyrics.

Method

Multilingual-BERT
- Multilingual version of the natural language processing model BERT developed by Google.
- Due to the characteristics of Korean songs, there are many cases where Korean and English are mixed, so I choose Multilingual-BERT.
- Embedded by line-by-line method.
- Experiment with the presence/absence of stopwords treatment.

DL Models
- Experiment with Five models.
- Batch size : 32, epochs : 10
