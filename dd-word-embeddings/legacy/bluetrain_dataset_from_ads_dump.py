# -*- coding: utf-8 -*-



'''


this script filters a dataset to match fa/fr metric gathering. more specifically, it:

1. Filters utterances with no stream id
2. Filters out non-Production user group utterances
3. Filters unwanted devices (currently empty blacklist)
4. Remove Alexa3P utterances with <tts> only transcriptions








# segment..utteranceId	stream..audio..class	stream..audio..isLiveTraffic	stream..audio..timestamp
stream..audio..locale	segment..ASRinfo..biasedTranscriptionSampling	segment..ASRinfo..supportedUtt
segment..ASRinfo..unsupportedUttType	segment..ASRinfo..supportedUttVersion	segment..ASRinfo..corrupted
segment..ASRinfo..hasBadTranscription	segment..ASRinfo..hasBadTranscriptionErrorType
segment..ASRinfo..hasBadTranscriptionRegexVersion	segment..ASRinfo..profaneFlag	segment..ASRinfo..profaneVersionDate
segment..ASRinfo..whitelist	segment..ASRinfo..supportedTransition	segment..ASRinfo..genderTransition
segment..ASRinfo..biasedAnnotationSampling	segment..ASRinfo..wwSupported	segment..ASRinfo..wwAccepted
segment..ASRinfo..mapping	segment..ASRinfo..askMode	segment..transcription..state	segment..transcription..verified
segment..transcription..gender	segment..transcription..nativity	segment..transcription..wakeWord
segment..transcription..convention	segment..transcription..content	segment..transcription..taskQueueId
segment..transcription..history..0..taskQueueId	segment..recognition..dialogAct	segment..recognition..nbest0
segment..recognition..conf0	segment..recognition..asrModelKey	segment..recognition..sdkModelKey
segment..recognition..sdkApplicationId	segment..recognition..sdkApplicationVersion	segment..recognition..audioInitiator
segment..nlu1best..domain	segment..nlu1best..intent	segment..nlu1best..interpretationScore	segment..nlu1best..text
segment..nlu1best..segmentationScore	segment..nlu1best..intentScore	segment..nlu1best..interpretationScoreBin
segment..annotation..domain	segment..annotation..intent	segment..annotation..description
segment..annotation..specVersion	segment..annotation..rawAnnotation	segment..sdkAnnotation..appId
segment..sdkAnnotation..content	segment..sdkAnnotation..intent	segment..sdkAnnotation..launchMode
segment..sdkAnnotation..state	segment..answerAnnotation..intent	segment..childDirectedRequest..deviceOpMode
segment..transcription..isNDonly	stream..ASRinfo..biasedAnnotationSampling	stream..ASRinfo..hasContextualLiterals
stream..ASRinfo..whitelist	stream..ASRinfo..SNR	stream..audio..audioDurationMillis	stream..audio..wwStartIndex
stream..audio..wwEndIndex	stream..audio..locale	stream..audio..speakerId	stream..audio..deviceId
stream..audio..deviceType	stream..audio..customerId	stream..audio..userGroup	stream..audio..deviceWW
stream..raw_audio..datamartRawAudioId	stream..audio..audioPlayerDeviceActivity
stream..audio..audioPlayerDeviceAlarmState	stream..audio..audioPlayerDeviceAudioPlayerState
stream..audio..audioPlayerDeviceBTAudioState	stream..audio..audioPlayerDeviceEarconPlayerState
stream..audio..audioPlayerDeviceTTSPlayerState	stream..audio..audioSamplingRateHertz


'''


import json
import re
import csv
import data_util
import ast

# weekrange = range(34, 41) # TRAIN
# weekrange = range(41, 43) # DEV
# weekrange = range(44, 45)  # TEST
#
exclude_wwo_wwi = False
exclude_empty = True

# if set to true, discards empty ASR transcriptions
trainMode = False
excludeDiscards = False

# data_type = 'test'
# data_out_loc = '/Users/kelleng/data/dd/ASI'
# dataset_type = 'ASI%s%s' % ('.with_WW' if not exclude_wwo_wwi else '', '.with_Empty' if not exclude_empty else '')
# dataset_name = '%s.%s.Week%d-%d' % (data_type, dataset_type, weekrange[0], weekrange[-1]) if len(weekrange) > 1 \
#     else '%s.%s.Week%d' % (data_type, data_type, weekrange[0])

# dataset_group = 'ASI.201809-201811'
dataset_group = 'Prod.v104'
dataset_type = 'test'

import dataset_groups
datasets = dataset_groups.groups[dataset_group][dataset_type]
newDatasetName = '%s.%s' % (dataset_group, dataset_type)

headers = None
utt_id_col = None
# stream_id_col = None
asr_result_col = None
transcription_col = None
nlu_intent_col = None
transcription_state_col = None
nd_only_col = None
user_group_col = None
device_type_col = None
customer_id_col = None
# dd_score_col = None


# filename = "/Users/kelleng/hover-workspace/bastion/user/all.index"
firstDataset = datasets[0]
if firstDataset.startswith('s3://'):
    firstDataset = data_util.download_from_s3(firstDataset)
with open(firstDataset, 'rb') as fdIn:
    for line in fdIn.readlines():
        try:
            # line = json.loads(line.decode('utf-8')).split('\t')
            line = line.split('\t')
            headers = line
            utt_id_col = headers.index('# segment..utteranceId')
            # stream_id_col = headers.index('stream..audio..streamId')
            asr_result_col = headers.index('segment..recognition..nbest0')
            transcription_col = headers.index('segment..transcription..content')
            nlu_intent_col = headers.index('segment..nlu1best..intent')
            transcription_state_col = headers.index('segment..transcription..state')
            nd_only_col = headers.index('segment..transcription..isNDonly')
            user_group_col = headers.index('stream..audio..userGroup')
            device_type_col = headers.index('stream..audio..deviceType')
            customer_id_col = headers.index('stream..audio..customerId')
            break
        except Exception as e:
            print e
            exit()


def collect_all_data():
    myDict = dict()
    myDictUtts = dict()
    nullDirStreams = list()
    passCount = 0
    # for weeknum in weekrange:
    # dumpname = 'Week%d/dump.dat' % weeknum
    # filename = '/Users/kelleng/workspace/paralinguisticsdataminer/src/ParalinguisticsDataMiner/aspan/notebook/wbr-data/' + dumpname
    # filename = "/Users/kelleng/hover-workspace/bastion/user/all.index"
    userGroupWhitelist = ['Production']
    deviceTypeBlacklist = []

    noStreamIdCount = 0
    wrongUserGroupCount = 0
    blacklistedDeviceCount = 0

    for dataset in datasets:
        if dataset.startswith('s3://'):
            dataset = data_util.download_from_s3(dataset)
        with open(dataset, 'rb') as fdIn:
            first = True
            for line in fdIn.readlines():
                try:
                    # line = json.loads(line.decode('utf-8')).split('\t')
                    line = line.split('\t')

                    if first:
                        first = False
                        continue

                    uttId = line[utt_id_col]
                    streamId = ''.join(uttId.split('/')[:-1])
                    asr_result = ast.literal_eval(line[asr_result_col].decode('utf-8'))
                    transcription = line[transcription_col]
                    nlu_intent = line[nlu_intent_col]

                    # Filter out no streamId
                    if streamId is None:
                        noStreamIdCount += 1
                        continue

                    # Filter out streams in wrong user group
                    if line[user_group_col] not in userGroupWhitelist:
                        wrongUserGroupCount += 1
                        continue

                    # Filter out blacklisted devices
                    if line[device_type_col] in deviceTypeBlacklist:
                        blacklistedDeviceCount += 1
                        continue


                    # # Filter out streams with null directedness
                    # if line['asr_directedness'] is None:
                    #     nullDirStreams.append(streamId)
                    #     continue

                    myDictUtts[uttId] = line
                    if streamId in myDict:
                        myDict[streamId]['utterances'].append(line)
                    else:
                        myDict[streamId] = {'stream_id': streamId, 'utterances': [line, ], 'asr_result': asr_result,
                                            'transcription': transcription, 'nlu_intent': nlu_intent}
                except Exception as e:
                    print e
                    # empty lines, etc.
                    passCount = passCount + 1
                    pass

    # print stats
    print 'No Stream ID skipped: ', noStreamIdCount
    print 'Wrong user group skipped: ', wrongUserGroupCount
    print 'Blacklisted device skipped: ', blacklistedDeviceCount

    # Sort utterances by utterance ID
    for streamId in myDict:
        myDict[streamId]['utterances'] = sorted(myDict[streamId]['utterances'], key=lambda x: x[utt_id_col])

    return myDict, myDictUtts, nullDirStreams, passCount


def getNDscore(uttBlob):
    if len(uttBlob['asr_result']) == 0:
        return [1, ]
    else:
        return [1 - 1.0 * uttBlob['asr_directedness'] / 1000, ]


def isWakeWordOnlySegment(segment):
    # from https://dp.amazon.com/display/BLUES/NLU+Cloud+WW+Verification
    if re.search(
            r'^(hi *|hey *|hello *|yeah *|ok *|okay *|はい *|ねえ *|ねぇ *|オーケー *)*(alexa *|amazon *|tango *|echo *|computer *|アレクサ *|エコー *|アマゾン *|コンピューター *)+$',
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
    uttList = streamBlob['utterances'][1:] if streamBlob['utterances'][0][transcription_state_col] == 'discard' else \
        streamBlob['utterances']
    # find the first utt that has non-wake-word ASR result
    for uttBlob in uttList:
        if len(uttBlob[asr_result_col]) == 0 or not isWakeWordOnlySegment(
                ' '.join([item[0] for item in uttBlob[asr_result_col]])):
            return uttBlob
    return None


def getRelevantUtt2(streamBlob):
    if len(streamBlob['utterances']) == 0: return None
    # ignore first utt if discard
    uttList = streamBlob['utterances'][1:] if streamBlob['utterances'][0][transcription_state_col] == 'discard' else streamBlob['utterances']
    #find the first utt that has non-wake-word ASR result
    for uttBlob in uttList:
        if len(uttBlob[asr_result_col]) != 0 and not isWakeWordOnlySegment(' '.join([item[0] for item in uttBlob[asr_result_col]])):
            return uttBlob
    return None



def getRelevantUtt_NLU_DD(streamBlob):
    utterances = streamBlob['utterances']

    if len(utterances) == 0:
        return None

    if len(utterances) == 1:
        return utterances[0]

    # ignore first if discarded
    utterances = utterances[1:] if utterances[0][transcription_state_col] == 'discard' else utterances

    # find the first utterance with non-empty, non-WW-only ASR result
    for utterance in utterances:
        asr_result = ast.literal_eval(utterance[asr_result_col].decode('utf-8'))
        if len(asr_result) != 0 and not isWakeWordOnlySegment(' '.join([item[0] for item in asr_result])):
            return utterance
    return None


# def fa_fr(streamList, myDict):
#     # Get TA utterances
#     taStreamList = list()
#     for streamId in streamList:
#         if myDict[streamId]['is_dd'] and myDict[streamId]['nd_score'][0] < 0.58:
#             taStreamList.append(streamId)
#             faTranscriptionList.append(
#                 (myDict[streamId]['asr_result'], myDict[streamId]['transcription'], myDict[streamId]['nlu_intent']))
#
#     # Get TR utterances
#     trStreamList = list()
#     for streamId in streamList:
#         if not myDict[streamId]['is_dd'] and myDict[streamId]['nd_score'][0] >= 0.58:
#             trStreamList.append(streamId)
#
#     # Get FA utterances
#     faStreamList = list()
#     for streamId in streamList:
#         if not myDict[streamId]['is_dd'] and myDict[streamId]['nd_score'][0] < 0.58:
#             faStreamList.append(streamId)
#
#     # Get FR utterances
#     frStreamList = list()
#     for streamId in streamList:
#         if myDict[streamId]['is_dd'] and myDict[streamId]['nd_score'][0] >= 0.58:
#             frStreamList.append(streamId)
#
#     # overallReportList = list()
#     # for thresh in [0.58, ]:
#     #     (overallReport, perStreamResultListSorted) = computeScores({x: [myDict[x]['is_dd'], ] for x in streamList},
#     #                                                                {x: [myDict[x]['nd_score'], ] for x in streamList},
#     #                                                                thresh, myDict)
#     #     print('%f   %f   %f   %f' % (
#     #     thresh, overallReport['errorRate'], overallReport['falseRejectRate'], overallReport['falseAlarmRate']))
#     #     print("------------")
#     #     overallReportList.append(overallReport)
#
#     print("Total utts:    %d" % len(uttList))
#     print("Total streams: %d" % len(streamList))
#     print("TA streams:    %d" % len(taStreamList))
#     print("TR streams:    %d" % len(trStreamList))
#     print("FA streams:    %d" % len(faStreamList))
#     print("FR streams:    %d" % len(frStreamList))
#     print("------------")


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
        if myDict[streamId]['utterances'][0][transcription_col] == '<tts>':
            tts3pRemovals.append(streamId)
            continue;

    selectedUtt = getRelevantUtt_NLU_DD(myDict[streamId])

    if selectedUtt is None:
        noRelevantUtts.append(streamId)
        continue

    selectedTransState = selectedUtt[transcription_state_col]
    selectedNDOnly = selectedUtt[nd_only_col]

    if selectedTransState == 'discard':
        myDict[streamId]['is_dd'] = False
        if selectedNDOnly == 'False':
            discardDisagreements.append(streamId)
    else:
        myDict[streamId]['is_dd'] = selectedNDOnly == 'False'

    # set stream-level asr_result and intent to the ground truth utterance asr_result and intent
    myDict[streamId]['asr_result'] = ast.literal_eval(selectedUtt[asr_result_col].decode('utf-8'))
    myDict[streamId]['nlu_intent'] = selectedUtt[nlu_intent_col]
    myDict[streamId]['transcription'] = selectedUtt[transcription_col]
    myDict[streamId]['transcription_state'] = selectedUtt[transcription_state_col]
    myDict[streamId]['selectedUtt'] = selectedUtt

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
    tts3pCustomers.add(myDict[streamId]['utterances'][0][customer_id_col])
    streamList.remove(streamId)
print("TTS 3P removals:  %d" % len(tts3pRemovals))
print("TTS 3P customers: %s" % str(tts3pRemovals))
print("------------")

faTranscriptionList = list()

# Compute utt list
uttList = set()
for streamId in streamList:
    for utt in myDict[streamId]['utterances']:
        uttList.add(utt[utt_id_col])

# fa_fr(streamList, myDict)

empty_asr_skipped = []
wwo_skipped = []
wwi_skipped = []
discards_skipped = []
with open('./%s.index' % newDatasetName, 'w') as out_file:
    # with open('%s/%s.stage1-results.csv' % (data_out_loc, dataset_name), 'w') as results_out_file:
    # out_file.write('%s\t%s\t%s\t%s\n' % ('uttId', 'intent', 'asr_result', 'isNDOnly'))
    out_file.write('\t'.join(headers))
    # results_writer = csv.writer(results_out_file)
    # results_writer.writerow(['uttid', 'directedness_score', 'directedness_target'])
    for streamId in streamList:
        # is_DD = 1 if myDict[streamId]['is_dd'] == True else 0
        is_NDOnly = not (myDict[streamId]['is_dd'] == True)
        # stage_1_score = myDict[streamId]['nd_score'][0]
        nlu_intent = myDict[streamId]['nlu_intent']
        asr_result = myDict[streamId]['asr_result']
        transcription_state = myDict[streamId]['transcription_state']

        if trainMode and len(asr_result) == 0:
            empty_asr_skipped.append(streamId)
            continue

        if excludeDiscards and transcription_state == 'discard':
            discards_skipped.append(streamId)
            continue

        if exclude_wwo_wwi and len(asr_result) == 1 and isWakeWordOnlySegment(asr_result[0][0]):
            wwo_skipped.append(streamId)
            continue

        if exclude_wwo_wwi and isWakeWordOnlySegment(asr_result[0][0]):
            wwi_skipped.append(streamId)
            continue

        # out_file.write('%s\t%s\t%s\t%s\n' % (streamId, nlu_intent, asr_result, is_NDOnly))
        out_file.write('\t'.join(myDict[streamId]['selectedUtt']))

print 'pre-filter', len(streamList)
print 'Empty ASR skipped', len(empty_asr_skipped)
print 'WW-only skipped', len(wwo_skipped)
print 'WW-initiated skipped', len(wwi_skipped)
print 'post-filter', len(streamList) - len(empty_asr_skipped) - len(wwo_skipped) - len(wwi_skipped)

# t_invalid = [None, '', '<ms>', '<<alexa>>']
# invalid = [None, '', '[]']

# Print FR streams
# print("FA streams:")
# blueshiftUrlFmt = 'https://blueshift-portal.amazon.com/jsp/utterance.html?%s'
# for faTranscription in faTranscriptionList:
#     if faTranscription[0] not in invalid and faTranscription[1] not in t_invalid and len(
#             faTranscription[0]) > 2 and "alexa" not in str(faTranscription[0]):
#         print(faTranscription[0])

# print("\n".join(faStreamList))