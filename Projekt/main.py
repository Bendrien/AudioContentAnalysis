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
    for drummer, drumName, template in data:

        # setup the file directories
        audioPath = u"../../ENST-drums-public/drummer_" + str(drummer) + "/audio/wet_mix/"
        annotationPath = u"../../ENST-drums-public/drummer_" + str(drummer) + "/annotation/"

        # get all audio file names
        #allAudioFiles = [f for f in listdir(audioPath) if path.isfile(path.join(audioPath, f)) and f.endswith(".wav")]

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
            result = findTemplate(audio, template, annotations, fs)

            # write results
            resultFilename = "../../Results/Drummer_" + str(drummer) + "/" + filename + "_" + drumName + ".txt"
            file = open(resultFilename, "w")
            file.write("[result, ground truth time, estimated time, difference]\n\n")
            file.write(str(result))
            file.close()

            # remember statistics
            positives.append(count_positives(result) / annotations[:, 0].size)
            false_positives.append((count_false_positives(result) + count_overflow(result)) / result[:, 0].size)
            trackNames.append(filename)

            # count analyzed tracks
            numberOfAnalyzedTracks += 1
            #if numberOfAnalyzedTracks > 1: break

        writeResults(drummer, drumName, numberOfAnalyzedTracks, positives, false_positives, trackNames)
        print("\n")



def writeResults(drummer, drumName, numberOfAnalyzedTracks, positives, false_positives, trackNames):
    filename = "../../Results/Drummer_" + str(drummer) + "/Results_Drummer_" + str(drummer) + "_" + drumName + ".md"
    print("Writing overall results to \"" + filename + "\"")
    file = open(filename, "w")

    file.write("# Results")
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


def findTemplate(audio, template, annotations, fs):
    frame = 1024
    hop = frame // 2

    # setup the W matrix with the template spectrum
    W1 = np.matrix(getMagnitudeSpectrum(template[0:frame]))
    W1 = W1.transpose()
    #print("W: " + str(W1.shape))

    # setup the v matrix with the audio spectrum
    V = np.matrix(blockwise(audio, frame, hop, getMagnitudeSpectrum))
    V = V.transpose()
    #print("V: " + str(V.shape))

    # setup the H matrix with random numbers
    H1 = np.matrix(np.random.rand(W1.shape[1], V.shape[1]))
    #print("H: " + str(H1.shape))

    # initialize the NMF
    lsnmf = nimfa.Nmf(V, W=W1, H=H1, seed=None, max_iter=20, rank=30)
    #lsnmf = nimfa.Nmf(V, seed=None, max_iter=20, rank=15)

    # do the NMF and get its results
    lsnmf_fit = lsnmf.factorize()
    #print("NMF iterations: " + str(lsnmf_fit.fit.n_iter))
    W = lsnmf_fit.basis()
    H = lsnmf_fit.coef()

    # uncomment the following block for a overview plot of the NMF matrices

    # f, axarr = plt.subplots(2, 2)
    # plt.subplots_adjust(hspace=0.5)
    #
    # axarr[0, 0].matshow(V, aspect='auto', origin='lower', norm=LogNorm(vmin=0.01, vmax=1))
    # axarr[0, 0].set_title('(a) Original spectrum')
    # axarr[0, 0].xaxis.set_ticks_position('bottom')
    # #axarr[0, 0].colorbar(im, cax=axcolor, ticks=t, format='$%.2f$')
    #
    # axarr[0, 1].matshow(W, aspect='auto', origin='lower')
    # axarr[0, 1].set_title('(b) W')
    # axarr[0, 1].xaxis.set_ticks_position('bottom')
    # # hacky way to get only 0 and 1 as x labels
    # axarr[0, 1].set_xlim([0, 0.5])
    # axarr[0, 1].set_xticklabels([0, 1], fontdict=None, minor=False)
    # axarr[0, 1].locator_params(axis='x', nbins=2)
    #
    # axarr[1, 0].matshow(H, aspect='auto', origin='lower')
    # axarr[1, 0].set_title('(c) H')
    # axarr[1, 0].xaxis.set_ticks_position('bottom')
    # # hacky way to get only 0 and 1 as y labels
    # axarr[1, 0].set_ylim([0, 0.5])
    # axarr[1, 0].set_yticklabels([0, 1], fontdict=None, minor=False)
    # axarr[1, 0].locator_params(axis='y', nbins=2)
    #
    # axarr[1, 1].matshow(W*H, aspect='auto', origin='lower', norm=LogNorm(vmin=0.01, vmax=1))
    # axarr[1, 1].set_title('(d) W*H')
    # axarr[1, 1].xaxis.set_ticks_position('bottom')
    #
    # plt.savefig("MatrixOverview.pdf", bbox_inches='tight')
    # plt.show()

    # convert H from a matrix to an normalized array
    Harray = H.A[0]
    Harray = Harray / max(Harray)

    thresholdline = np.empty_like(Harray)
    thresholdline.fill(2/3)

    # plt.figure()
    # plt.plot(thresholdline, color='r')
    # plt.plot(Harray)
    # plt.savefig("ActivationMatrix.pdf", bbox_inches='tight')
    # plt.show()

    # apply threshold
    Harray = np.maximum(Harray, 2/3)
    Harray = np.diff(Harray)

    # plt.figure()
    # plt.plot(Harray)
    # plt.xlabel("STFT frame number")
    # plt.savefig("ZeroCrosses.pdf", bbox_inches='tight')
    # plt.show()

    #Harray[Harray == 0] = np.nan
    #zeroCrossings = np.where(np.diff(np.signbit(Harray)))[0]
    zeroCrossings = []
    for x in range(1, len(Harray)):
        if np.signbit(Harray[x - 1]) < np.signbit(Harray[x]):
            zeroCrossings.append(x)

    times = (np.array(zeroCrossings) * hop / fs)
    data = np.array(list(map(lambda t: ['bd', t], times)))


    epsilon = hop / fs
    return compare_times(annotations[:, 1], data[:, 1], epsilon)

    # synthAudio = []
    # delta = V
    # for row in delta:
    #     row = np.squeeze(np.asarray(row))  # transform 1D matrix to array
    #     synthAudio.extend(np.fft.ifft(row))
    #
    # plt.figure()
    # plt.plot(synthAudio)
    #
    # mx = 32767
    # synthAudio = np.fromiter((s * mx for s in synthAudio), dtype=np.int16)
    # wavio.write("synth.wav", synthAudio, fs)


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


class Result:

    # There are more detected events then annotated
    UNDERFLOW = -2

    # A detetcted value lies under its annotation.
    LOWER = -1

    # A value equals its corresponding annotation
    EQUAL = 0

    # A detetcted value lies over its annotation.
    UPPER = 1

    # There are more annotated events then detetcted
    OVERFLOW = 2


# NOTE: For this function to work both datasets (groundtruth and data) need to be in ascending order!
def compare_times(groundtruth, data, epsilon):
    # [result, gt_time, data_time, diff]
    results = []

    groundtruth_iter = iter(groundtruth)
    data_iter = iter(data)
    data_time = next(data_iter)

    # simultaneously advance the ground truth and the data iterator
    for gt_time in groundtruth_iter:
        while True:
            result, diff = compare(gt_time, data_time, epsilon)

            # if its the first result
            if len(results) == 0:
                # just append it
                results.append([result, gt_time, data_time, diff])

            else:
                # compare with previous result
                prev_cmp, prev_gt_time, prev_data_time, prev_diff = results[-1]

                # if the previous result was a match
                if prev_cmp == Result.EQUAL:
                    # just append the current one
                    results.append([result, gt_time, data_time, diff])

                # if the accurancy of the previous result wasn't as good as the current one (then two not corresponding values where compared)
                elif diff < prev_diff:
                    # just pop the last result because its not meaningful.
                    results.pop()
                    results.append([result, gt_time, data_time, diff])

            # if the result was marked as LOWER
            if result == Result.LOWER:
                # just advance the ground truth iterator by leaving the inner loop
                break

            # try to advance the data iterator
            try:
                # if we got a match
                if result == Result.EQUAL:
                    # advance the data iterator
                    data_time = next(data_iter)
                    # also advance the ground truth iterator by leaving the inner loop
                    break

                # if the result was marked as UPPER
                elif result == Result.UPPER:
                    # just advance the data iterator
                    data_time = next(data_iter)

            # if we couldn't advance the data iterator (because its empty)
            except StopIteration as e:
                # check for all left values in the ground truth iterator
                for gt_time in groundtruth_iter:
                    results.append([Result.UNDERFLOW, gt_time, -1, -1])
                return np.array(results)

    # are some values left in the data iterator?
    for data_time in data_iter:
        results.append([Result.OVERFLOW, -1, data_time, -1])

    return np.array(results)


def compare(a, b, epsilon):
    diff = float(a) - float(b)
    diff_abs = abs(diff)

    if diff_abs <= epsilon:
        return Result.EQUAL, diff_abs
    if diff < 0.0:
        return Result.LOWER, diff_abs

    return Result.UPPER, diff_abs



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

if __name__ == '__main__':
    main()
