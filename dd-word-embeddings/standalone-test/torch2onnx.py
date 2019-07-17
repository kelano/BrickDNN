import torch
from torch.autograd import Variable
import torch.onnx as torch_onnx
import simple_lstm_decoupled

import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
from mxnet.test_utils import download

import onnx


# Load the pytorch model
# model_path = "./SimpleLSTM_Decoupled"
model_path = '/Users/kelleng/Desktop/trained-models/SimpleLSTM_Decoupled'

model = simple_lstm_decoupled.SimpleLSTMDecoupled(
    in_size=300,
    hidden_size=150,
    out_size=2,
    batch_size=1,
    lstm_layers=1,
    use_cuda=False)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

model.summary()

# Export the model to an ONNX file
input_shape = (1, 10, 300)
dummy_input = Variable(torch.randn(*input_shape))
# print dummy_input

onnx_model_file = "test.onnx"

output = torch_onnx.export(model, dummy_input, onnx_model_file, verbose=True)

print("Export of torch_model.onnx complete!")

sym, arg, aux = onnx_mxnet.import_model(onnx_model_file)

print("Import complete!")