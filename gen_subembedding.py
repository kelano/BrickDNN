import csv
import io

all_words = []

with open('data-updated.csv', 'r') as csvfile:
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


fin = io.open('..\wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
fout = io.open('..\wiki-news-300d-1M-subset.vec', 'w', encoding='utf-8', newline='\n', errors='ignore')

fout.write(fin.readline())
for line in fin:
    tokens = line.rstrip().split(' ')
    word = tokens[0]
    if word in unique_words or word == 'UNK':
        fout.write(line)
