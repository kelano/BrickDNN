import data_util
import torch
import torch.nn.functional as F
from models.baseline_bow_mlp import BOWMLP
import dd_platform


def run_eval_utt(sentence, word_2_embedding, model):
    model.eval()
    x = torch.Tensor(data_util.get_bow_encoding(sentence, word_2_embedding))
    out = model(x)
    _, pred_label = torch.max(out.data, 1)
    dd_pred = pred_label.data[0].item() == 0
    out = F.softmax(out, dim=1)
    dd_conf = out.data[0, 0]
    print 'DD Conf: %f Pred: %s' % (dd_conf.item(), dd_pred)


# model_path = 'BOW-MLP'
model_path = 'BOW-MLP-MixedTrain'
model_name = model_path.split('/')[-1]

model = BOWMLP(300, 2, 600)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

word_2_embedding = data_util.load_embedding_as_dict('%s/embeddings/wiki-news-300d-1M-subset.vec' % dd_platform.DATA_LOC)

while True:
    s = raw_input('--> ')
    run_eval_utt(s, word_2_embedding, model)
