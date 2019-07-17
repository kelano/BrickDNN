
data_loc = '/Users/kelleng/data/dd'
# data_loc = '/Users/kelleng/hover-workspace/repos/Djinn/work/kelleng/dd-word-embeddings/dd-data'
# data_loc = '/home/ec2-user/workspaces/hover-workspace/repos/Djinn/work/kelleng/dd-word-embeddings/dd-data'
# data_loc = '/home/ec2-user/dd-data'

# for suffix in ('.index',):
for suffix in ('.stage1-results.csv',):

    to_combine = [
        '%s/ADS/test.ADS.Week43-44%s' % (data_loc, suffix),
        '%s/prodv1/test%s' % (data_loc, suffix)
    ]
    
    combined_name = '%s/mixed/test.prodv1_ADS.Week43-44%s' % (data_loc, suffix)
    
    header_added = False
    with open(combined_name, 'wb') as outfile:
        for index in to_combine:
            index_first = True
            with open(index, 'r') as infile:
                for line in infile:
                    if index_first:
                        if not header_added:
                            outfile.write(line)
                            header_added = True
                        index_first = False
                    else:
                        outfile.write(line)

