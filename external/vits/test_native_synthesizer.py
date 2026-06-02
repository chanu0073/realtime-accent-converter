import torch

from native_synthesizer import NativeSynthesizer


model = NativeSynthesizer(
    n_vocab=200,

    spec_channels=80,

    segment_size=32,

    inter_channels=192,
    hidden_channels=192,

    filter_channels=768,

    n_heads=2,
    n_layers=6,

    kernel_size=3,
    p_dropout=0.1,

    resblock="1",

    resblock_kernel_sizes=[3,7,11],

    resblock_dilation_sizes=[
        [1,3,5],
        [1,3,5],
        [1,3,5]
    ],

    upsample_rates=[8,8,2,2],

    upsample_initial_channel=512,

    upsample_kernel_sizes=[
        16,16,4,4
    ],

    gin_channels=256
)

print("Model created")


B = 2

text = torch.randint(
    0,
    200,
    (B,50)
)

text_lengths = torch.LongTensor(
    [50,45]
)

mel = torch.randn(
    B,
    80,
    400
)

mel_lengths = torch.LongTensor(
    [400,380]
)

f0 = torch.randn(
    B,
    1,
    400
)

outputs = model(
    text,
    text_lengths,
    mel,
    mel_lengths,
    f0
)

print("\nForward successful\n")

for i,o in enumerate(outputs):

    if isinstance(o, torch.Tensor):
        print(i, o.shape)

    elif isinstance(o, tuple):
        print(i, "tuple")

    else:
        print(i, type(o))
