import csv
import io


def gen_subembedding(fname, foutname, all_words, insert_pad_token=True):
    # load full embedding
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
            print(int(float(count) / n * 100), "% complete")
        count += 1




all_words = []
all_words.append('UNK')

with open('data_201903.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    first = True
    data = []
    for row in reader:
        if first:
            # ignore, header
            first = False
            headers = row
        else:
            data.append(row)
    # print headers
    # print data[0]

    filtered_data = []
    name_index = headers.index('Name')
    for data_row in data:
        name = data_row[name_index]
        for word in name.split():
            all_words.append(word.lower())

unique_words = set(all_words)

print(len(all_words), len(unique_words))
# exit()


# fin = io.open('../wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
# fout = io.open('../wiki-news-300d-1M-subset.vec', 'w', encoding='utf-8', newline='\n', errors='ignore')

gen_subembedding('../wiki-news-300d-1M.vec', '../wiki-news-300d-1M-subset.vec', unique_words, insert_pad_token=True)

# fout.write(fin.readline())
# for line in fin:
#     tokens = line.rstrip().split(' ')
#     word = tokens[0]
#     if word in unique_words or word == 'UNK':
#         fout.write(line)
