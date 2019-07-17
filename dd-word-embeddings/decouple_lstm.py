"""
Utility to decouple the SimpleLSTM model type, removing the embedding layer to be used separately at runtime.
"""


import data_util
import trained_model_groups
import torch


# embeddings
embedding = trained_model_groups.models['Prod.v100.Local']['SimpleLSTM_150H']['embedding']
# embedding = 's3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M-subset-v104.vec'
embedding_mat, word_2_idx = data_util.load_embedding_as_numpy_mat(embedding)
word_2_embedding = data_util.load_embedding_as_dict(embedding)

model, conf = data_util.create_model_instance('Prod.v100.Local', 'SimpleLSTM_150H', embedding_mat)
model_path = trained_model_groups.models['Prod.v100.Local']['SimpleLSTM_150H']["loc"]
state_dict = torch.load(model_path, map_location='cpu')
print state_dict
model.load_state_dict(state_dict)

model = data_util.load_trained_model('Prod.v100.Local', 'SimpleLSTM_150H', embedding_mat=embedding_mat)

print model

for x in model.children():
    print x


from models.simple_lstm_decoupled import SimpleLSTMDecoupled
decoupledModel = SimpleLSTMDecoupled(
            in_size=300,
            hidden_size=conf["HIDDEN_SIZE"],
            out_size=2,
            batch_size=conf["BATCH_SIZE"],
            lstm_layers=conf["LSTM_LAYERS"],
            use_cuda=False)

print decoupledModel
decoupledModelStateDict = decoupledModel.state_dict()
newStateDict = {}
for name, param in state_dict.items():
    if name in decoupledModelStateDict:
        print 'importing ',name
        newStateDict[name] = param
    else:
        print 'unable to import ',name
decoupledModel.load_state_dict(newStateDict)

torch.save(decoupledModel.state_dict(), 'SimpleLSTM_Decoupled')
