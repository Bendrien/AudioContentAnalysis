import numpy as np
import glob
import wavio
import matplotlib.pyplot as plt
import math
import copy


def main():
    hop = 512
    frame = 1024

    samples = []
    for filename in glob.glob('samples/training/perc/*.wav'):
        fs, audio = getNormalizedAudio(filename)
        samples.append([fs, audio, -1])

    for filename in glob.glob('samples/training/tonal/*.wav'):
        fs, audio = getNormalizedAudio(filename)
        samples.append([fs, audio, 1])

    # Aufgabe 1
    meta = analyze(samples, frame, hop)
    KNN = normalize(meta)

    fig, ax = plt.subplots()

    for s in KNN[1]:
        if s[3] == -1:
            marker = None
        else:
            marker = 's'

        ax.scatter(s[0], s[1], color='r', marker=marker, alpha=.6)
        ax.scatter(s[1], s[2], color='g', marker=marker, alpha=.6)
        ax.scatter(s[0], s[2], color='b', marker=marker, alpha=.6)

    plt.show()

    # Aufgabe 2
    featureString = ["Rolloff", "Centroid", "Flux", "Nothing"]
    map = {'sax.wav':1, 'Cello.wav':1, 'cowb.wav':-1, 'hitom.wav':-1}
    testSamples = []
    for filename in glob.glob('samples/testset/*.wav'):
        fs, audio = getNormalizedAudio(filename)
        testSamples.append([fs, audio, filename[16:]])

    testMeta = analyze(testSamples, frame, hop)

    for i in range(0, len(featureString)):
        print(featureString[i] + " ignored:")
        for s in testMeta:
            klasse = classify(copy.deepcopy(s), KNN, 3, i)
            name = s[3]
            isCorrect = map[name] == klasse
            if isCorrect:
                print(name + " is classified as " + getClassName(klasse) + ", correct!")
            else:
                print(name + " is classified as " + getClassName(klasse) + ", wrong :(")
        print()

    # Aufgabe 3
    first = 0
    second = 2
    fig, ax = plt.subplots()

    for s in KNN[1]:
        if s[3] == -1:
            marker = None
        else:
            marker = 's'

        ax.scatter(s[first], s[second], color='r', marker=marker, alpha=.6)

    for s in testMeta:
        s[first] /= KNN[0][first]
        s[second] /= KNN[0][second]
        ax.scatter(s[first], s[second], color='g', marker='x', alpha=.6)

    plt.show()

    return


def analyze(samples, frameSize, hopSize):
    meta = []
    for fs, sample, label in samples:
        rolloff = blockwise(sample, frameSize, hopSize, spectralRolloff, [fs])
        centroid = blockwise(sample, frameSize, hopSize, spectralCentroid, [fs])
        flux = blockwise(sample, frameSize, hopSize, spectralFlux)
        meta.append([np.mean(rolloff), np.mean(centroid), np.var(flux), label])
    return meta


def classify(sn, knn, k, ignore):
    if len(sn) > 3:
        sn.pop(-1)
    sn = np.array(sn)

    # normalize with std
    sn /= knn[0]

    # remove ignored feature
    if ignore < 3:
        sn = np.delete(sn, ignore)

    distances = []
    for i, elem in enumerate(knn[1]):
        a = np.array(elem)
        # remove label
        a = np.delete(a, -1)
        # remove ingored feature
        if ignore < 3:
            a = np.delete(a, ignore)
        dist = np.linalg.norm(a - sn)
        distances.append([i, dist])

    distances.sort(key=lambda x: x[1])

    klasse = 0
    for i, _ in distances[0:k]:
        klasse += knn[1][int(i)][3]

    if klasse > 0:
        return 1
    else:
        return -1


def getClassName(klasse):
    if klasse > 0:
        return "tonal"
    else:
        return "percussive"


def normalize(v):
    out = list(zip(*v))
    stds = []
    for i, dimension in enumerate(out):
        if i < 3:
            std = np.std(dimension)
            out[i] /= std
            stds.append(std)

    return [np.array(stds), list(zip(*out))]



def getNormalizedAudio(filename: str):
    # read in audio and convert it to normalized floats
    wav = wavio.read(filename)
    audio = wav.data
    fs = wav.rate
    maxVal = np.max(np.abs(audio))
    audio = np.fromiter((s / maxVal for s in audio), dtype=float)
    return fs, audio


#########################################
### Functions from task 3
#########################################


# Calculates the spectral rolloff
# Returns the frequency below which 95% of the energy is located
def spectralRolloff(frame, previousFrame, fs: int):
    magnitudeSpectrum = getMagnitudeSpectrum(frame)
    thresholdSum = 0.95 * sum(i * i for i in magnitudeSpectrum)

    rolloffIndex = 0
    rolloffSum = 0.0
    for i in range(0, len(magnitudeSpectrum)):
        rolloffSum += magnitudeSpectrum[i] * magnitudeSpectrum[i]
        if rolloffSum >= thresholdSum:
            rolloffIndex = i
            break

    return rolloffIndex / len(magnitudeSpectrum) * (fs / 2)  # TODO: Frage: Frequenzberechnung nicht in Formel


# Calculates the spectral centroid
def spectralCentroid(frame, previousFrame, fs: int):
    magnitudeSpectrum = getMagnitudeSpectrum(frame)
    frameSize = len(frame)
    # positive center frequencies for each bin
    frequencies = np.abs(np.fft.fftfreq(frameSize, 1.0/fs)[:frameSize // 2 + 1])  # TODO: Frage: Frequenzberechnung nicht in Formel
    return sum(frequencies * magnitudeSpectrum) / sum(magnitudeSpectrum)


# Calculates the spectral flux
def spectralFlux(frame, previousFrame):
    magnitudeSpectrum = getMagnitudeSpectrum(frame)
    previousMagnitudeSpectrum = getMagnitudeSpectrum(previousFrame)
    K = len(frame)

    return sum(np.square(np.subtract(magnitudeSpectrum, previousMagnitudeSpectrum))) / K


# x:            Audiosamples (list like)
# fs:           Sample rate
# frameSize:    Frame size in sample
# hopSize:      Hop size in sample
# filename:     Name of the file
def blockwiseFeature(x, fs: int, frameSize: int, hopSize: int, filename: str):
    rolloff = blockwise(x, frameSize, hopSize, spectralRolloff, [fs])
    centroid = blockwise(x, frameSize, hopSize, spectralCentroid, [fs])
    flux = blockwise(x, frameSize, hopSize, spectralFlux)

    f, subplots = plt.subplots(3)
    plt.subplots_adjust(hspace=0.5)
    subplots[0].plot(rolloff)
    subplots[0].set_title('Spectral rolloff der Datei ' + filename)
    subplots[0].set_xlabel("Frames")
    subplots[0].set_ylabel("Frequenz (Hz)")
    subplots[1].plot(centroid)
    subplots[1].set_title('Spectral centroid der Datei ' + filename)
    subplots[1].set_xlabel("Frames")
    subplots[1].set_ylabel("Frequenz (Hz)")
    subplots[2].plot(flux)
    subplots[2].set_title('Spectral flux der Datei ' + filename)
    subplots[2].set_xlabel("Frames")
    subplots[2].set_ylabel("Frequenz (Hz)")
    plt.show()


#########################################
### Helper functions
#########################################


def msToSamples(timeInMs: int, fs: int):
    return int(math.ceil(timeInMs / 1000 * fs))


def getMagnitudeSpectrum(frame):
    return abs(np.fft.rfft(frame))


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
    previousFrame = np.zeros(frameSize)

    if zeroPad:
        numberOfFrames -= 1  # minus 1, because we zero padded

    # blockwise processing
    for frameCount in range(0, numberOfFrames):
        begin = int(frameCount * hopSize)
        frame = x[begin: begin + frameSize]
        output.append(blockwiseFunction(frame, previousFrame, *functionArgs))
        previousFrame = frame

    return output

if __name__ == '__main__':
    main()