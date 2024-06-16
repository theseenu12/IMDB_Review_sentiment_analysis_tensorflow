from msilib import Feature
import tensorflow as tf
from tensorflow import keras, train
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split
import IPython
from IPython import display
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix,multilabel_confusion_matrix
import json
import io


# print(tf.__version__)

# (x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data()

# word_index = keras.datasets.imdb.get_word_index()

# word_index = dict((values,key) for (key,values) in word_index.items())

# print(len(x_train))
# print('__________')
# print(len(y_train))

# print(len(x_train[0]))

features_dataset = keras.preprocessing.text_dataset_from_directory('train/')


vocab_size = 10000
max_length = 120
embedding_dim = 16
trunc_type = 'post'


tokenizer = keras.preprocessing.text.Tokenizer(vocab_size,oov_token='<OOV>')


# print(features_padded[:2])

# print(labels[:2])


features = []

labels = []

for fea,lab in features_dataset.unbatch().as_numpy_iterator():
    features.append(str(fea))

    labels.append(lab)
    
# print(features[:2])

# print(labels[:2])


features = np.array(features,dtype=str)

labels = np.array(labels)
    

model = keras.Sequential([keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
                          keras.layers.Flatten(),
                          keras.layers.Dense(6,activation='relu'),
                          keras.layers.Dense(1,activation='sigmoid')                          
                          ])


print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

tokenizer.fit_on_texts(features)

word_index = tokenizer.word_index

wc = tokenizer.word_counts

nw = tokenizer.num_words

features_sequence = tokenizer.texts_to_sequences(features)

# print(features[:2])

features_padded = keras.utils.pad_sequences(features_sequence,maxlen=max_length,padding='post')

print(features_padded[:2])

print(len(word_index))
# print(word_index)
# print(wc)
# print(nw)

# wwe = ['how are you doing','what the hell you are doing','what happened to you','come here and say hello']

# wwe_token = keras.preprocessing.text.Tokenizer(10,oov_token='<OOV>')

# wwe_token.fit_on_texts(wwe)

# wwe_sequence = wwe_token.texts_to_sequences(wwe)

# wwe_padding = keras.utils.pad_sequences(wwe_sequence,maxlen=10,padding='post')

# print(wwe_padding)

# wwe_wi = wwe_token.word_index

# wwe_wc = wwe_token.word_counts

# wwe_nw = wwe_token.num_words

# word_docs = wwe_token.word_docs

# print(wwe_wi)

# print(wwe_wc)
# print(wwe_nw)
# print(word_docs)

reversed_word_index = dict([(values,key) for (key,values) in word_index.items()])

# texts = tokenizer.sequences_to_texts(features_sequence)

# print(texts[:2])


def decode_review(padded_text):
    return " ".join(reversed_word_index.get(i,'?') for i in padded_text)


decode_features_pad = decode_review(features_padded[10])

print("Decoded")
print(decode_features_pad)
print(labels[10])

print(len(features_padded))


hist = model.fit(features_padded,labels,batch_size=32,epochs=10,verbose=1,validation_split=0.2)

plt.plot(hist.history['val_loss'],c='r')
plt.plot(hist.history['loss'],c='g')

plt.legend(['val_loss','train_loss'])

plt.show()

e = model.layers[0]

weights = np.array(e.get_weights()[0])

print(weights.shape)

out_v = io.open('vets.tsv','w',encoding='utf-8')
out_m = io.open('meta.tsv','w',encoding='utf-8')


for word_num in range(1,vocab_size):
    
    word = reversed_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join(str(x) for x in embeddings) + '\n')
    
out_v.close()
out_m.close()

## using lstm

# model2 = keras.Sequential([keras.layers.Embedding(vocab_size,64,input_length=max_length),
#                            keras.layers.Bidirectional(keras.layers.LSTM(64)),
#                           keras.layers.Dense(64,activation='relu'),
#                           keras.layers.Dense(1,activation='sigmoid')
#                           ])

# model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# print(model2.summary())
# hist2 = model2.fit(features_padded,labels,batch_size=32,epochs=10,verbose=1,validation_split=0.2)

# plt.plot(hist2.history['loss'])
# plt.plot(hist2.history['val_loss'])
# plt.legend(['loss','val_loss'])

# plt.show()

## using conv1d 

# model3 =    tf.keras.Sequential([
#                     tf.keras.layers.Embedding(vocab_size,64,input_length=max_length),
#                     tf.keras.layers.Conv1D(128, 5, activation='relu'),
#                     tf.keras.layers.GlobalAveragePooling1D(),
#                     tf.keras.layers.Dense(64, activation='relu'),
#                     tf.keras.layers.Dense(1, activation='sigmoid')
                    
#                     ])

# model3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# hist3 = model3.fit(features_padded,labels,batch_size=32,epochs=10,verbose=1,validation_split=0.2)

# plt.plot(hist3.history['loss'])
# plt.plot(hist3.history['val_loss'])
# plt.legend(['loss','val_loss'])

# plt.show()


        

        
        
    