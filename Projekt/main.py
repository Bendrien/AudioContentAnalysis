import pydub as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import csv
import nimfa

from os import listdir
from os import path


def main():

    # load the drum templates
    fs, sdTemplate = getNormalizedAudio("001_hits_snare-drum_sticks.wav")
    fs, bdTemplate = getNormalizedAudio("004_hits_bass-drum_pedal.wav")
    fs, ohhTemplate = getNormalizedAudio("005_hits_open-hi-hat_sticks.wav")
    fs, chhTemplate = getNormalizedAudio("006_closed-hi-hat_sticks.wav")
    fs, rc3Template = getNormalizedAudio("008_hits_ride-cymbal-1_sticks.wav")
    fs, cbTemplate = getNormalizedAudio("013_hits_cowbell_sticks.wav")

    # plt.plot(template)
    # plt.axvspan(0, 1024, color='red', alpha=0.5)
    # plt.xlabel("Sample")
    # plt.ylabel("Amplitude")
    # plt.savefig("BassdrumTemplate.pdf")
    # plt.show()

    # Prepare the data
    # data = [[2, "sd", sdTemplate]]
    data = [[2, "bd", bdTemplate]]
    # data = [[2, "ohh", ohhTemplate]]
    # data = [[2, "chh", chhTemplate]]
    # data = [[2, "rc3", rc3Template]]
    # data = [[2, "cb", cbTemplate]]
    #data = [[1, "bd", bdTemplate], [2, "bd", bdTemplate], [3, "bd", bdTemplate]]
    #data = [[1, "chh", chhTemplate], [2, "chh", chhTemplate], [3, "chh", chhTemplate]]

    for drummer, drumName, template in data:

        # setup the file directories
        audioPath = u"../../ENST-drums-public/drummer_" + str(drummer) + "/audio/wet_mix/"
        annotationPath = u"../../ENST-drums-public/drummer_" + str(drummer) + "/annotation/"

        # get all audio file names
        allAudioFiles = [f for f in listdir(audioPath) if path.isfile(path.join(audioPath, f)) and f.endswith(".wav")]

        # TO BE REMOVED: test track for the plots
        allAudioFiles = ["045_phrase_rock_simple_medium_sticks.wav"]

        print("Transcripting: " + drumName + " of drummer " + str(drummer) + "\n=============")

        positives = []
        false_positives = []
        trackNames = []
        numberOfAnalyzedTracks = 0

        for audioFile in allAudioFiles:

            # load annotations
            filename = path.splitext(audioFile)[0]
            annotations = loadAnnotationFile(annotationPath + filename + ".txt", drumName)

            # check if the file contains the drum sound
            if annotations.size == 0:
                # otherwise skip the track
                continue

            print("Analyzing track: " + filename)

            # load the audio data
            fs, audio = getNormalizedAudio(audioPath + audioFile)

            # find the drum sound
            result = findTemplate(audio, template, annotations, fs, drumName)

            # remember statistics
            positives_count = count_positives(result)
            false_positives_count = count_false_positives(result) + count_overflow(result)
            annotation_count = annotations[:, 0].size
            result_count = result[:, 0].size
            writeTrackResults(drummer, filename, drumName, annotation_count,
                              positives_count, false_positives_count, result)

            positives.append(positives_count / annotation_count)
            false_positives.append(false_positives_count / result_count)
            trackNames.append(filename)

            # count analyzed tracks
            numberOfAnalyzedTracks += 1
            #if numberOfAnalyzedTracks >= 2: break

        writeOverallResults(drummer, drumName, numberOfAnalyzedTracks, positives, false_positives, trackNames)
        print("\n")



# find the template in the given audio and compare it to the annotations
# returns a list of all compare
def findTemplate(audio, template, annotations, fs, drumName):
    frame = 1024
    hop = frame // 2    # 50% overlap

    threshold = 2 / 3
    epsilon = hop / fs


    # setup the W matrix with the template spectrum
    W1 = np.matrix(getMagnitudeSpectrum(template[0:frame]))
    W1 = W1.transpose()

    # setup the v matrix with the audio spectrum
    V = np.matrix(blockwise(audio, frame, hop, getMagnitudeSpectrum))
    V = V.transpose()

    # setup the H matrix with random numbers
    H1 = np.matrix(np.random.rand(W1.shape[1], V.shape[1]))

    # initialize the NMF
    lsnmf = nimfa.Nmf(V, W=W1, H=H1, seed=None, max_iter=20, rank=30)

    # do the NMF and get its results
    lsnmf_fit = lsnmf.factorize()
    W = lsnmf_fit.basis()
    H = lsnmf_fit.coef()

    # convert H from a matrix to an normalized array
    Harray = H.A[0]
    Harray = Harray / max(Harray)

    # apply threshold
    Harray = np.maximum(Harray, threshold)
    Harray = np.diff(Harray)

    # compute zero crossings
    zeroCrossings = []
    for x in range(1, len(Harray)):
        if np.signbit(Harray[x - 1]) < np.signbit(Harray[x]):
            zeroCrossings.append(x)

    # generate time estimation
    times = (np.array(zeroCrossings) * hop / fs)
    data = np.array(list(map(lambda t: [drumName, t], times)))

    # compare estimations and annotations
    result = compare_times(annotations[:, 1], data[:, 1], epsilon)


    # uncomment the following plot blocks for the various figures in the paper

    # f, axarr = plt.subplots(2, 2)
    # plt.subplots_adjust(hspace=0.5)
    #
    # axarr[0, 0].matshow(V, aspect='auto', origin='lower', norm=LogNorm(vmin=0.01, vmax=1))
    # axarr[0, 0].set_title('(a) Original spectrum')
    # axarr[0, 0].xaxis.set_ticks_position('bottom')
    # axarr[0, 0].set_xlabel("STFT frame number")
    # axarr[0, 0].set_ylabel("Frequency bin")
    # #axarr[0, 0].colorbar(im, cax=axcolor, ticks=t, format='$%.2f$')
    #
    # axarr[0, 1].matshow(W, aspect='auto', origin='lower')
    # axarr[0, 1].set_title('(b) W')
    # axarr[0, 1].xaxis.set_ticks_position('bottom')
    # # hacky way to get only 0 and 1 as x labels
    # axarr[0, 1].set_xlim([0, 0.5])
    # axarr[0, 1].set_xticklabels([0, 1], fontdict=None, minor=False)
    # axarr[0, 1].locator_params(axis='x', nbins=2)
    # axarr[0, 1].set_xlabel("Template")
    # axarr[0, 1].set_ylabel("Frequency bin")
    #
    # axarr[1, 0].matshow(H, aspect='auto', origin='lower')
    # axarr[1, 0].set_title('(c) H')
    # axarr[1, 0].xaxis.set_ticks_position('bottom')
    # # hacky way to get only 0 and 1 as y labels
    # axarr[1, 0].set_ylim([0, 0.5])
    # axarr[1, 0].set_yticklabels([0, 1], fontdict=None, minor=False)
    # axarr[1, 0].locator_params(axis='y', nbins=2)
    # axarr[1, 0].set_xlabel("STFT frame number")
    # axarr[1, 0].set_ylabel("Template")
    #
    # axarr[1, 1].matshow(W*H, aspect='auto', origin='lower', norm=LogNorm(vmin=0.01, vmax=1))
    # axarr[1, 1].set_title('(d) W*H')
    # axarr[1, 1].xaxis.set_ticks_position('bottom')
    # axarr[1, 1].set_xlabel("STFT frame number")
    # axarr[1, 1].set_ylabel("Frequency bin")
    #
    # plt.savefig("MatrixOverview.pdf", bbox_inches='tight')
    # plt.show()

    # plt.figure()
    # thresholdline = np.empty_like(Harray)
    # thresholdline.fill(threshold)
    # plt.plot(thresholdline, color='r')
    # plt.plot(Harray)
    # plt.xlabel("STFT frame number")
    # plt.ylabel("Magnitude")
    # plt.savefig("ActivationMatrix.pdf", bbox_inches='tight')
    # plt.show()

    # plt.figure()
    # plt.plot(Harray)
    # plt.xlabel("STFT frame number")
    # plt.savefig("ZeroCrosses.pdf", bbox_inches='tight')
    # plt.show()

    return result


# datatype of the time comparing result
class Result:
    # There are more annotated events then detetcted
    UNDERFLOW = -2

    # A detetcted value lies under its annotation.
    LOWER = -1

    # A value equals its corresponding annotation
    EQUAL = 0

    # A detetcted value lies over its annotation.
    UPPER = 1

    # There are more detected events then annotated
    OVERFLOW = 2


def compare(a, b, epsilon):
    diff = float(a) - float(b)
    diff_abs = abs(diff)

    if diff_abs <= epsilon:
        return Result.EQUAL, diff_abs
    if diff < 0.0:
        return Result.LOWER, diff_abs

    return Result.UPPER, diff_abs


# returns a list of all compare ordered like this:
# [[result, groundtruth_time, data_time, diff], ...]
def compare_times(groundtruth, data, epsilon):
    results = []

    groundtruth_iter = iter(groundtruth)
    data_iter = iter(data)

    groundtruth_time = next(groundtruth_iter)
    data_time = next(data_iter)

    # advance both iterators side by side as needed
    # NOTE: for this to work the times in both datasets (groundtruth and data) need to be in ascending order!
    try:
        while True:
            result, diff = compare(groundtruth_time, data_time, epsilon)
            apply_result = result

            # check for duplicates
            if results:
                prev_result, prev_groundtruth_time, prev_data_time, prev_diff = results[-1]

                # if a groundtruth_time appears more then once we have an overflow (false positive)
                if prev_groundtruth_time == groundtruth_time:
                    # label the less precise one as OVERFLOW
                    if diff < prev_diff:
                        results[-1][0] = Result.OVERFLOW
                    else:
                        apply_result = Result.OVERFLOW

                # if the same data_time appears multiple times in the results we have an underflow
                if prev_data_time == data_time:
                    # label the less precise one as UNDERFLOW
                    if diff < prev_diff:
                        results[-1][0] = Result.UNDERFLOW
                    else:
                        apply_result = Result.UNDERFLOW

            results.append([apply_result, groundtruth_time, data_time, diff])

            # advance the iterators
            if result == Result.EQUAL:
                groundtruth_time = next(groundtruth_iter)
                data_time = next(data_iter)
            elif result == Result.LOWER:
                groundtruth_time = next(groundtruth_iter)
            elif result == Result.UPPER:
                data_time = next(data_iter)
            else:
                print("Warning: undefined behaviour in compare_times!")
                break

    # if one or both iterators can't be advanced anymore the loop will stop
    except:
        # check for left values in the ground truth iterator
        for time in groundtruth_iter:
            results.append([Result.UNDERFLOW, time, -1, -1])

        # check for left values in the data iterator
        for time in data_iter:
            results.append([Result.OVERFLOW, -1, time, -1])

    return np.array(results)


#########################################
### Helper functions
#########################################


def count_positives(data):
    return np.count_nonzero(data[:,0] == str(Result.EQUAL))


def count_false_positives(data):
    neg = np.count_nonzero(data[:,0] == str(Result.LOWER))
    pos = np.count_nonzero(data[:,0] == str(Result.UPPER))
    return neg + pos


def count_underflow(data):
    return np.count_nonzero(data[:,0] == str(Result.UNDERFLOW))


def count_overflow(data):
    return np.count_nonzero(data[:,0] == str(Result.OVERFLOW))


def getMagnitudeSpectrum(frame):
    return abs(np.fft.rfft(frame))


def getNormalizedAudio(filename: str):
    audio = pd.AudioSegment.from_wav(filename)

    # downmix to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)

    samples = np.array(audio.get_array_of_samples())
    fs = pd.utils.mediainfo(filename)['sample_rate']

    # normailze to range [-1, 1]
    maxVal = np.iinfo(samples.dtype).max
    samples = np.fromiter((s / maxVal for s in samples), dtype=float)

    return int(fs), samples


# x:                 Audio samples (list like)
# frameSize:         Frame size in sample
# hopSize:           Hop size in sample
# blockwiseFunction: The function that is to be called blockwise (first function parameter is the block)
# functionArgs:      Additional argument the the blockwise function
# zeroPad:           Enabled zero padding at the end to fit the frameSize
def blockwise(x, frameSize: int, hopSize: int, blockwiseFunction, functionArgs=None, zeroPad: bool = True):

    if functionArgs is None:
        functionArgs = []

    if zeroPad:
        # zero pad at the end
        rest = len(x) % hopSize
        if rest != 0:
            remainder = frameSize - rest
        else:
            remainder = hopSize
        x = np.lib.pad(x, (0, remainder), 'constant', constant_values=0)

    # calculate the number of blocks
    numberOfFrames = int(math.ceil(len(x) / hopSize))
    output = []

    if zeroPad:
        numberOfFrames -= 1  # minus 1, because we zero padded

    # blockwise processing
    for frameCount in range(0, numberOfFrames):
        begin = int(frameCount * hopSize)
        frame = x[begin: begin + frameSize]
        output.append(blockwiseFunction(frame, *functionArgs))

    return output


def loadAnnotationFile(filename, selector=None):
    if not isinstance(selector, str):
        selector = None

    with open(filename) as csvfile:
        result = []
        reader = csv.reader(csvfile, delimiter=" ")
        for row in reader:
            if selector is not None and row[1] != selector:
                continue
            result.append(np.array([row[1], float(row[0])]))

    return np.array(result)


def writeTrackResults(drummer, filename, drumName, annotation_count, positives_count, false_positives_count, result):
    resultFilename = "../../Results/Drummer_" + str(drummer) + "/" + filename + "_" + drumName + ".txt"
    file = open(resultFilename, "w")
    file.write("# Track Results")
    file.write("\nThere are " + str(result[:, 0].size) + " result entries compared to " + str(annotation_count)
               + " annotations in track \"" + filename + "\".")
    file.write("\nPositives:\t" + str(positives_count))
    file.write("\nFalse positives:\t" + str(false_positives_count))

    file.write("\n\n## Result legend")
    file.write("\nList entries:\n\t[RESULT, annotated time, estimated time, difference]")
    file.write("\nRESULT mapping:")
    file.write("\n\tUNDERFLOW\t= -2\tEstimation doesn't have a corresponding annotation")
    file.write("\n\tLOWER\t\t= -1\tEstimation lies beneath its annotation")
    file.write("\n\tEQUAL\t\t=  0\tEstimation equals its annotation")
    file.write("\n\tUPPER\t\t=  1\tEstimation lies above its annotation")
    file.write("\n\tOVERFLOW\t=  2\tAnnotation doesn't have a corresponding Estimation")

    file.write("\n\n## Entries\n")
    file.write(str(result))
    file.close()


def writeOverallResults(drummer, drumName, numberOfAnalyzedTracks, positives, false_positives, trackNames):
    filename = "../../Results/Drummer_" + str(drummer) + "/Results_Drummer_" + str(drummer) + "_" + drumName + ".md"
    print("Writing overall results to \"" + filename + "\"")
    file = open(filename, "w")

    file.write("# Overall Results")
    file.write("\nDrummer:\t" + str(drummer))
    file.write("\nDrumsound:\t" + str(drumName))
    file.write("\nNumber of analyzed files: " + str(numberOfAnalyzedTracks))
    file.write("\n")

    file.write("\n## Positives")
    file.write("\nMedian:\t" + str(np.median(positives)))
    file.write("\nMean:\t" + str(np.mean(positives)))
    file.write("\nStandard Deviation:\t" + str(np.std(positives)))
    file.write("\n")

    file.write("\n## False Positives")
    file.write("\nMedian:\t" + str(np.median(false_positives)))
    file.write("\nMean:\t" + str(np.mean(false_positives)))
    file.write("\nStandard Deviation:\t" + str(np.std(false_positives)))
    file.write("\n")

    file.write("\n## Positives Data\n")
    file.write(str(np.array(sorted(zip(positives, trackNames)))))

    file.write("\n")
    file.write("\n## False Positives Data\n")
    file.write(str(np.array(sorted(zip(false_positives, trackNames)))))
    file.close()


if __name__ == '__main__':
    main()
