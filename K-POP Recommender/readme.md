
"K-POP Title Recommendation Based on Multilingual-BERT"

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
- LSTM : When the input sequence becomes too long, RNN has a gradient vanishing or exploding problem when processing it. To address this, the LSTM is designed. LSTM uses two vectors to store a short-term state (ht) and a long-term state (ct), respectively. The key to LSTM is to learn what the network will remember, what to delete, and what to read in the long-term state (ct). This solves the problem of long-term dependence!
- GRU : GRU is a simplified version of the complex LSTM model, the two state vectors ct and ht in LSTM Cell were merged into one vector ht. Although it has the same structure as LSTM, it has fewer parameters than LSTM, so the computational cost is low, and the learning speed is faster, but it is a model that performs similarly.
- BiLSTM : Processing information in both directions, better understanding of context is possible. Double the hidden dimension during the conversion from hidden dimension to output dimension because it is two-way. The core concept of two-way LSTMs is to add another LSTM that runs from back to front (reverse) on the last node, rather than proceeding with learning only in the forward direction. Bi-LSTM adds a hidden layer that conveys information in the reverse direction from the general LSTM, so that at each point in time, the hidden state has the effect of having information from both the previous point in time and the future point in time.
