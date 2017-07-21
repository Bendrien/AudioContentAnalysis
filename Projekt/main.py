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

    # load the bass drum template
    fs, template = getNormalizedAudio("base_drum.wav")

    # plt.plot(template)
    # plt.axvspan(0, 1024, color='red', alpha=0.5)
    # plt.savefig("BassdrumTemplate.pdf")
    # plt.show()

    # setup the file directories
    drummer = 2
    audioPath = u"../../ENST-drums-public/drummer_" + str(drummer) + "/audio/wet_mix/"
    annotationPath = u"../../ENST-drums-public/drummer_" + str(drummer) + "/annotation/"

    # get all audio file names
    allAudioFiles = [f for f in listdir(audioPath) if path.isfile(path.join(audioPath, f)) and f.endswith(".wav")]
    #allAudioFiles = ["045_phrase_rock_simple_medium_sticks.wav"]

    positives = []
    false_positives = []

    trackNames = []
    results = []
    numberOfAnalyzedTracks = 0
    for audioFile in allAudioFiles:
        # load annotations
        filename = path.splitext(audioFile)[0]
        annotations = loadAnnotationFile(annotationPath + filename + ".txt", "bd")

        # check if the files contains bass drums
        if annotations.size == 0:
            # otherwise skip the track
            continue

        # load the audio data
        fs, audio = getNormalizedAudio(audioPath + audioFile)

        # find the bass drums
        print("Analyzing track: " + filename)
        result = findTemplate(audio, template, annotations, fs)
        positives.append(count_positives(result) / annotations[:, 0].size)
        false_positives.append((count_false_positives(result) + count_overflow(result)) / result[:, 0].size)
        trackNames.append(filename)
        results.append(result)

        file = open("../../Results/Drummer_" + str(drummer) + "/" + filename + ".txt", "w")
        file.write(str(result))
        file.close()

        numberOfAnalyzedTracks += 1
        #if numberOfAnalyzedTracks > 1: break

    file = open("../../Results/Drummer_" + str(drummer) + "/Results_" + str(drummer) + ".md", "w")

    file.write("# Results\n")
    file.write("\n")

    file.write("Number of analyzed files: " + str(numberOfAnalyzedTracks))
    file.write("\n")

    file.write("\n### Positives")
    file.write("\nMedian:\t" + str(np.median(positives)))
    file.write("\nMean:\t" + str(np.mean(positives)))
    file.write("\nStandart Deviation:\t" + str(np.std(positives)))
    file.write("\n")

    file.write("\n### False Positives")
    file.write("\nMedian:\t" + str(np.median(false_positives)))
    file.write("\nMean:\t" + str(np.mean(false_positives)))
    file.write("\nStandart Deviation:\t" + str(np.std(false_positives)))
    file.write("\n")

    file.write("\n### Positives Data\n")
    file.write(str(np.array(sorted(zip(positives, trackNames)))))

    file.write("\n")
    file.write("\n### False Positives Data\n")
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

    # f, axarr = plt.subplots(2, 2)
    # plt.subplots_adjust(hspace=0.5)
    #
    # axarr[0, 0].matshow(V, aspect='auto', origin='lower', norm=LogNorm(vmin=0.01, vmax=1))
    # axarr[0, 0].set_title('Original spectrum')
    # axarr[0, 0].xaxis.set_ticks_position('bottom')
    # #axarr[0, 0].colorbar(im, cax=axcolor, ticks=t, format='$%.2f$')
    # print("V: " + str(V.shape))
    #
    # axarr[0, 1].matshow(W, aspect='auto', origin='lower')
    # axarr[0, 1].set_title('W')
    # axarr[0, 1].xaxis.set_ticks_position('bottom')
    # print("W: " + str(W.shape))
    #
    # axarr[1, 0].matshow(H, aspect='auto', origin='lower')
    # axarr[1, 0].set_title('H')
    # axarr[1, 0].xaxis.set_ticks_position('bottom')
    # print("H: " + str(H.shape))
    #
    # axarr[1, 1].matshow(W*H, aspect='auto', origin='lower', norm=LogNorm(vmin=0.01, vmax=1))
    # axarr[1, 1].set_title('W*H')
    # axarr[1, 1].xaxis.set_ticks_position('bottom')
    #
    # plt.savefig("MatrixOverview.pdf")
    # plt.show()

    # convert H from a matrix to an normalized array
    Harray = H.A[0]
    Harray = Harray / max(Harray)

    thresholdline = np.empty_like(Harray)
    thresholdline.fill(2/3)

    # plt.figure()
    # plt.plot(thresholdline, color='r')
    # plt.plot(Harray)
    # plt.savefig("ActivationMatrix.pdf")
    # plt.show()

    # apply threshold
    Harray = np.maximum(Harray, 2/3)
    Harray = np.diff(Harray)

    # plt.figure()
    # plt.plot(Harray)
    # plt.savefig("ZeroCrosses.pdf")
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
    return np.count_nonzero(data[:,0] == '0')


def count_false_positives(data):
    neg = np.count_nonzero(data[:,0] == '1')
    pos = np.count_nonzero(data[:,0] == '-1')
    return neg + pos

def count_underflow(data):
    return np.count_nonzero(data[:,0] == '2')

def count_overflow(data):
    return np.count_nonzero(data[:,0] == '3')


class Compare:
    LOWER = -1
    EQUAL = 0
    UPPER = 1

def compare_times(groundtruth, data, epsilon):
    # [cmp, gt_time, data_time, diff]
    result = []

    groundtruth_iter = iter(groundtruth)
    data_iter = iter(data)
    data_time = next(data_iter)

    for gt_time in groundtruth_iter:
        while True:
            cmp, diff = compare(gt_time, data_time, epsilon)

            if len(result) == 0:
                result.append([cmp, gt_time, data_time, diff])
            else:
                prev_cmp, prev_gt_time, prev_data_time, prev_diff = result[-1]
                if prev_cmp == Compare.EQUAL:
                    result.append([cmp, gt_time, data_time, diff])
                elif diff < prev_diff:
                    result.pop()
                    result.append([cmp, gt_time, data_time, diff])

            if cmp == Compare.LOWER:
                break

            try:
                if cmp == Compare.EQUAL:
                    data_time = next(data_iter)
                    break
                elif cmp == Compare.UPPER:
                    data_time = next(data_iter)
            except StopIteration as e:
                # are some left in the groundtrouth?
                for gt_time in groundtruth_iter:
                    result.append([2, gt_time, -1, -1])
                return np.array(result)


    # are some left in the data?
    for data_time in data_iter:
        result.append([3, -1, data_time, -1])
    return np.array(result)


def compare(a, b, epsilon):
    diff = float(a) - float(b)
    diff_abs = abs(diff)

    if diff_abs <= epsilon:
        return Compare.EQUAL, diff_abs
    if diff < 0.0:
        return Compare.LOWER, diff_abs

    return Compare.UPPER, diff_abs



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
