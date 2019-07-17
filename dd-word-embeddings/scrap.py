import json
import data_util
import trained_model_groups
import torch
import re


intent = 'SportsQAIntent'
intent2 = 'QAIntentSports'
intent3 = 'GetWeatherIntent'

# search = 'QAIntent'
search = '^QAIntent'


print re.search(search, intent) == None
print re.search(search, intent2) == None
print re.search(search, intent3) == None
exit()


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


torch.save(model.state_dict(), 'ModelStateDict')

exit()


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

exit()


with open("./configs/bow_mlp.json") as json_data_file:
    conf = json.load(json_data_file)
    print conf



import ast

u1 = "[]"
u2 = "[[u'turn', 995], [u'off', 1000], [u'the', 986], [u'fan', 966], [u'light', 679]]"

# print type(u1.decode('utf-8'))
# print u2.decode('utf-8')
#
# print ast.literal_eval(u1.decode('utf-8'))
res = ast.literal_eval(u2.decode('utf-8'))





print int(True)
print int(False)

print res, len(res), len(res[0])



class Data:
    class v104:
        def __init__(self):
            self.__name__ = 'v104'
        test='something'
        train='okay'
        dev='whatever'
    class ASI:
        test='something'





print Data.v104.test, Data.v104


exit()


print ast.literal_eval(u1)
print ast.literal_eval(u2)

asr_result = json.loads(u1.decode('utf-8'))
asr_result = json.loads(u2.decode('utf-8'))