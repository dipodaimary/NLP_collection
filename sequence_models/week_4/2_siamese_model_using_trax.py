import trax
from trax import layers as tl
import trax.fastmath.numpy as np
import numpy

def normalize(x):
    return x/np.sqrt(np.sum(x*x, axis=-1, keepdims=True))

'''
Siamese Model
'''
vocab_size = 500
model_dimension = 128

# define the LSTM model
LSTM = tl.Serial(
    tl.Embedding(vocab_size=vocab_size, d_feature=model_dimension),
    tl.LSTM(model_dimension),
    tl.Mean(axis=1),
    tl.Fn('Normalize', lambda x: normalize(x))
)

Siamese = tl.Parallel(LSTM, LSTM)

def show_layers(model, layer_prefix):
    print(f"Total layers: {len(model.sublayers)}")
    for i in range(len(model.sublayers)):
        print("========")
        print(f"{layer_prefix}_{i}: {model.sublayers[i]}")

show_layers(Siamese, 'Parallel.sublayers')

show_layers(LSTM, 'Serial.sublayers')


