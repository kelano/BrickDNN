# -*- coding: utf-8 -*-

import json
import re
import csv


# weekrange = range(34, 41) # TRAIN
# weekrange = range(41, 43) # DEV
weekrange = range(34, 45) # ALL

exclude_wwo_wwi = False
exclude_empty = False
data_type = 'ALL'
data_out_loc = '/Users/kelleng/data/dd/ADS'
dataset_type = 'ADS%s%s' % ('.with_WW' if not exclude_wwo_wwi else '', '.with_Empty' if not exclude_empty else '')
dataset_name = '%s.%s.Week%d-%d' % (data_type, dataset_type, weekrange[0], weekrange[-1]) if len(weekrange) > 1\
    else '%s.%s.Week%d' % (data_type, data_type, weekrange[0])


def collect_all_data():
    myDict = dict()
    myDictUtts = dict()
    nullDirStreams = list()
    passCount = 0
    for weeknum in weekrange:
        dumpname = 'Week%d/dump.dat' % weeknum
        filename = '/Users/kelleng/workspace/paralinguisticsdataminer/src/ParalinguisticsDataMiner/aspan/notebook/wbr-data/' + dumpname
        userGroupWhitelist = ['Production']
        deviceTypeBlacklist = []
    
        with open(filename, 'rb') as fdIn:
            for line in fdIn.readlines():
                try:
                    line = json.loads(line.decode('utf-8'))
                    uttId = line['utterance_id']
                    streamId = line['stream_id']
                    asr_result = line['asr_result']
                    transcription = line['transcription']
                    nlu_intent = line['nlu_intent']
                    
                    # Filter out no streamId
                    if streamId is None:
                        continue
        
                    # Filter out streams in wrong user group
                    if line['user_group'] not in userGroupWhitelist:
                        continue
        
                    # Filter out blacklisted devices
                    if line['device_type'] in deviceTypeBlacklist:
                        continue
        
                    # Filter out streams with null directedness
                    if line['asr_directedness'] is None:
                        nullDirStreams.append(streamId)
                        continue
        
                    myDictUtts[uttId] = line
                    if streamId in myDict:
                        myDict[streamId]['utterances'].append(line)
                    else:
                        myDict[streamId] = {'stream_id': streamId, 'utterances': [line, ], 'asr_result': asr_result,
                                            'transcription': transcription, 'nlu_intent': nlu_intent}
                except:
                    # empty lines, etc.
                    passCount = passCount + 1
                    pass

    # Sort utterances by utterance ID
    for streamId in myDict:
        myDict[streamId]['utterances'] = sorted(myDict[streamId]['utterances'], key=lambda x: x['utterance_id'])

    return myDict, myDictUtts, nullDirStreams, passCount


def getNDscore(uttBlob):
    if len(uttBlob['asr_result']) == 0:
        return [1, ]
    else:
        return [1 - 1.0 * uttBlob['asr_directedness'] / 1000, ]


def isWakeWordOnlySegment(segment):
    # from https://dp.amazon.com/display/BLUES/NLU+Cloud+WW+Verification
    if re.search(r'^(hi *|hey *|hello *|yeah *|ok *|okay *|はい *|ねえ *|ねぇ *|オーケー *)*(alexa *|amazon *|tango *|echo *|computer *|アレクサ *|エコー *|アマゾン *|コンピューター *)+$',
                 segment.strip().lower()) != None:
        return True
    # this one is to capture single leading words, such as "ummm alexa"
    elif re.search(r'^(\w)+ (alexa *|amazon *|tango *|echo *|computer *|アレクサ *|エコー *|アマゾン *|コンピューター *)+$',
                   segment.strip().lower()) != None:
        return True
    else:
        return False


def getRelevantUtt(streamBlob):
    if len(streamBlob['utterances']) == 0: return None
    # ignore first utt if discard
    uttList = streamBlob['utterances'][1:] if streamBlob['utterances'][0]['transcription_state'] == 'discard' else \
    streamBlob['utterances']
    # find the first utt that has non-wake-word ASR result
    for uttBlob in uttList:
        if len(uttBlob['asr_result']) == 0 or not isWakeWordOnlySegment(
                ' '.join([item[0] for item in uttBlob['asr_result']])):
            return uttBlob
    return None


def fa_fr(streamList, myDict):
    # Get TA utterances
    taStreamList = list()
    for streamId in streamList:
        if myDict[streamId]['is_dd'] and myDict[streamId]['nd_score'][0] < 0.58:
            taStreamList.append(streamId)
            faTranscriptionList.append(
                (myDict[streamId]['asr_result'], myDict[streamId]['transcription'], myDict[streamId]['nlu_intent']))

    # Get TR utterances
    trStreamList = list()
    for streamId in streamList:
        if not myDict[streamId]['is_dd'] and myDict[streamId]['nd_score'][0] >= 0.58:
            trStreamList.append(streamId)

    # Get FA utterances
    faStreamList = list()
    for streamId in streamList:
        if not myDict[streamId]['is_dd'] and myDict[streamId]['nd_score'][0] < 0.58:
            faStreamList.append(streamId)

    # Get FR utterances
    frStreamList = list()
    for streamId in streamList:
        if myDict[streamId]['is_dd'] and myDict[streamId]['nd_score'][0] >= 0.58:
            frStreamList.append(streamId)

    # overallReportList = list()
    # for thresh in [0.58, ]:
    #     (overallReport, perStreamResultListSorted) = computeScores({x: [myDict[x]['is_dd'], ] for x in streamList},
    #                                                                {x: [myDict[x]['nd_score'], ] for x in streamList},
    #                                                                thresh, myDict)
    #     print('%f   %f   %f   %f' % (
    #     thresh, overallReport['errorRate'], overallReport['falseRejectRate'], overallReport['falseAlarmRate']))
    #     print("------------")
    #     overallReportList.append(overallReport)

    print("Total utts:    %d" % len(uttList))
    print("Total streams: %d" % len(streamList))
    print("TA streams:    %d" % len(taStreamList))
    print("TR streams:    %d" % len(trStreamList))
    print("FA streams:    %d" % len(faStreamList))
    print("FR streams:    %d" % len(frStreamList))
    print("------------")


myDict, myDictUtts, nullDirStreams, passCount = collect_all_data()





print(str(len(nullDirStreams)) + " null directedness removals")
print(str(passCount) + " passes")
print(str(passCount) + " passes")
print()
print("streams: " + str(len(myDict)))
print("utts: " + str(len(myDictUtts)))


streamList = list(myDict.keys())
streamListLeft = list()
noRelevantUtts = list()
tts3pRemovals = list()
discardDisagreements = list()
for streamId in myDict:
    # TEMP: Remove 3p streams with "thank you" TTS picked up
    if streamId.startswith("Alexa3P"):
        if myDict[streamId]['utterances'][0]['transcription'] == '<tts>':
            tts3pRemovals.append(streamId)
            continue;

    if len(myDict[streamId]['utterances']) == 0:
        raise ValueError('unexpected')
    elif len(myDict[streamId]['utterances']) == 1 and myDict[streamId]['utterances'][0][
        'transcription_state'] == 'discard':
        streamListLeft.append(streamId)
        # if there's only one junk utternaces in a stream, the stream is considered nd
        uttBlob = myDict[streamId]['utterances'][0]
        myDict[streamId]['is_dd'] = 0
        myDict[streamId]['nd_score'] = getNDscore(uttBlob)
    elif len(myDict[streamId]['utterances']) == 1 and myDict[streamId]['utterances'][0][
        'transcription_state'] != 'discard':
        uttBlob = myDict[streamId]['utterances'][0]
        myDict[streamId]['is_dd'] = uttBlob['is_nd_groundtruth'] == False
        myDict[streamId]['nd_score'] = getNDscore(uttBlob)
    elif len(myDict[streamId]['utterances']) == 2 and myDict[streamId]['utterances'][0][
        'transcription_state'] == 'discard':
        uttBlob = myDict[streamId]['utterances'][1]
        myDict[streamId]['is_dd'] = uttBlob['is_nd_groundtruth'] == False
        myDict[streamId]['nd_score'] = getNDscore(uttBlob)
    else:
        uttBlob = getRelevantUtt(myDict[streamId])
        if uttBlob is None:
            # myPrint(myDict, num=100, subList=[streamId,])
            noRelevantUtts.append(streamId)
            continue;
        if uttBlob['transcription_state'] == 'discard':
            myDict[streamId]['is_dd'] = 0
            if uttBlob['is_nd_groundtruth'] == False:
                discardDisagreements.append(streamId)
        else:
            myDict[streamId]['is_dd'] = uttBlob['is_nd_groundtruth'] == False
        myDict[streamId]['nd_score'] = getNDscore(uttBlob)
    
    # set stream-level asr_result and intent to the ground truth utterance asr_result and intent
    myDict[streamId]['asr_result'] = uttBlob['asr_result']
    myDict[streamId]['nlu_intent'] = uttBlob['nlu_intent']
    myDict[streamId]['transcription'] = uttBlob['transcription']

# Remove streams with no relevant utts
for streamId in noRelevantUtts:
    streamList.remove(streamId)
print("No relevant utt removals: %d" % len(noRelevantUtts))
print("------------")

# TEMP: print discard disagreements
print("Discard disagreements = %d" % len(discardDisagreements))

# TEMP: remove 3p TTS streams
tts3pCustomers = set()
for streamId in tts3pRemovals:
    tts3pCustomers.add(myDict[streamId]['utterances'][0]['customer_id'])
    streamList.remove(streamId)
print("TTS 3P removals:  %d" % len(tts3pRemovals))
print("TTS 3P customers: %s" % str(tts3pRemovals))
print("------------")

# Compute utt list
uttList = set()
for streamId in streamList:
    for utt in myDict[streamId]['utterances']:
        uttList.add(utt['utterance_id'])


with open('%s/%s_streams' % (data_out_loc, dataset_name), 'w') as out_file:
    for stream in streamList:
        out_file.write('%s\n' % stream)

with open('%s/%s_utts' % (data_out_loc, dataset_name), 'w') as out_file:
    for utt in uttList:
        out_file.write('%s\n' % utt)