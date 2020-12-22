import trax
from trax import layers as tl

mlp = tl.Serial(
    tl.Dense(128),
    tl.Relu(),
    tl.Dense(10),
    tl.LogSoftmax()
)

print(mlp)

''' 
GRU Model
To create a GRU model you need to be familiar with the following

1) ShiftRight: shifts the tensor to the right by padding on axis 1.
The mode should be specified and it refers to the context in 
which the model is being used. Possible values are: 'train', 'eval'
or 'predict', predict mode is for fast inference. Defaults to
'train'.

2) Embedding: maps discrete tokens to vectors. It will have shape
(vocabulary length X dimension of output vectors). The dimension
of output vectors (also called d_features) is the number of 
elements in the word embedding.

3) GRU: The GRU layer. It leverages another trax layer call GRUCell.
The number of GRU units should be specified and should match the 
number of elements in the word embedding. If you want to stack two 
consecutive GRU layers, it can be done by using python's list 
comprehension.

4) Dense: vanilla dense layer

5) LogSoftMax: log softmax function. 
'''


mode = 'train'
vocab_size = 256
model_dimension = 512
n_layers = 2

GRU = tl.Serial(
    tl.ShiftRight(mode=mode), # do remember to pass the mode parameter, default is train
    tl.Embedding(vocab_size=vocab_size, d_feature=model_dimension),
    [tl.GRU(n_units=model_dimension) for _ in range(n_layers)],
    tl.Dense(n_units=vocab_size),
    tl.LogSoftmax()
)

# helper function that prints information for every layer (sublayer with Serial)
def show_layers(model, layer_prefix="Serial.sublayers"):
    print(f"Total layers: {len(model.sublayers)}\n")
    for i in range(len(model.sublayers)):
        print("=============")
        print(f"{layer_prefix}_{i}: {model.sublayers[i]}\n")

show_layers(GRU)


