
# "K-POP Title Recommendation Based on Multilingual-BERT"

## Introduction
- The application of deep learning technology is increasingly expanding to content generation such as image, text, and music.
- Increasing interest in the field of automatically learning music styles using different corpora and creating new music content based on them.
" Development of a model that predicts and recommends song titles by analyzing text data of song lyrics"

There is no model learned as Korean lyrics, there are many studies on lyrics generation, but there is a lack of research on title generation.

In addition, methods used in lyrics generation studies are mainly limited to RNN or GRU.

In this project, various model tests other than RNN and GRU are performed.

## Dataset

2000-2023 Oct. Melon chart 100 dataset

From : https://github.com/EX3exp/Kpop-lyric-datasets

We extract data from the original data in json format from 2010 to October 2023.

It consists of 18,000 songs and 900,000 lyrics.

## Method

### Multilingual-BERT
- Multilingual version of the natural language processing model BERT developed by Google.
- Due to the characteristics of Korean songs, there are many cases where Korean and English are mixed, so I choose Multilingual-BERT.
- Embedded by line-by-line method.
- Experiment with the presence/absence of stopwords treatment.

### DL Models
- Experiment with Five models.
- Batch size : 32, epochs : 10

### LSTM
- When the input sequence becomes too long, RNN has a gradient vanishing or exploding problem when processing it.
- To address this, the LSTM is designed. LSTM uses two vectors to store a short-term state (ht) and a long-term state (ct), respectively.
- The key to LSTM is to learn what the network will remember, what to delete, and what to read in the long-term state (ct). This solves the problem of long-term dependence!
```
class LyricsClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LyricsClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(label_encoder.classes_)

model_L = LyricsClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_L.parameters(), lr=0.001)
```

### GRU
- GRU is a simplified version of the complex LSTM model, the two state vectors ct and ht in LSTM Cell were merged into one vector ht.
- Although it has the same structure as LSTM, it has fewer parameters than LSTM, so the computational cost is low, and the learning speed is faster, but it is a model that performs similarly.
```
class LyricsGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LyricsGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, seq_length, input_dim)
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        out = self.fc(gru_out)
        return out

input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(label_encoder.classes_)

model_G = LyricsGRU(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_G.parameters(), lr=0.001)
```
- BiLSTM : Processing information in both directions, better understanding of context is possible. Double the hidden dimension during the conversion from hidden dimension to output dimension because it is two-way. The core concept of two-way LSTMs is to add another LSTM that runs from back to front (reverse) on the last node, rather than proceeding with learning only in the forward direction. Bi-LSTM adds a hidden layer that conveys information in the reverse direction from the general LSTM, so that at each point in time, the hidden state has the effect of having information from both the previous point in time and the future point in time.
- Transformer : Transformers can parallelize long sequence data, showing excellent performance in natural language processing tasks (translation, summarization, Q&A, etc.) Using the Self-Attention mechanism, all words in the input sequence can learn relationships with each other, enabling parallel processing, leading to improved training speed and performance.
- CNN : NN extracts image features from multiple convolutional and pooling layers, classify images through a fully connected layer. Learning is possible while maintaining spatial information in the image. It ia a neural network structure that is effective for learning spatial features of images.

Results
