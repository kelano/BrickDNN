"""
Tool for generating embedding subset for a given set of data."
"""


import data_util
import io
import dataset_groups


def gen_subembedding(fname, foutname, all_words, insert_pad_token=True):
    # load full embedding
    if fname.startswith("s3://"):
        fname = data_util.download_from_s3(fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    # calculate new N for subset embedding heading
    new_n = 1 if insert_pad_token else 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in all_words:
            new_n += 1

    # re-open full embedding
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fout = io.open(foutname, 'w', encoding='utf-8', newline='\n', errors='ignore')

    # write new header
    fout.write(u'%d %d\n' % (new_n, d))

    # write pad token if applicable
    if insert_pad_token:
        new_n += 1
        fout.write(u'<PAD> ')
        fout.write(u' '.join([u'0.0' for i in range(d)]))
        fout.write(u'\n')

    # write all vectors present in dataset tokens
    count = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in all_words:
            fout.write(line)
        if count % 10000 == 0:
            print int(float(count) / n * 100), "% complete"
        count += 1


groups = [
    # 'Prod.v104',
    'Prod.v107',
    'ASI.201809-201812',
]

all_words = set()
all_words.add('unk')

for dataset_group in groups:
    for dataset_name in ['train', 'dev', 'test']:
        print 'processing ', dataset_group, dataset_name
        dataset = dataset_groups.groups[dataset_group][dataset_name]
        utt_2_intent, utt_2_sentence, utt_2_confidences, utt_2_truth = data_util.get_hover_datasets(dataset)
        for sentence in utt_2_sentence.values():
            sentence = data_util.preprocess_sentence(sentence)
            for word in sentence.split():
                all_words.add(word)
            
print '%d unique tokens found' % len(all_words)


gen_subembedding('s3://bluetrain-workspaces/kelleng/dd-data/embeddings/wiki-news-300d-1M.vec',
                 './wiki-news-300d-1M-subset.vec', all_words, insert_pad_token=True)

