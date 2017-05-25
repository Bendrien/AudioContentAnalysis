import numpy as np
import scipy.io.wavfile
from scipy import signal
import matplotlib.pyplot as plt
import math


# Calculates the spectral rolloff
# Returns the frequency below which 95% of the energy is located
def spectralRolloff(frame, previousFrame, fs: int):
    magnitudeSpectrum = getMagnitudeSpectrum(frame)
    percent = 0.95
    thresholdSum = percent * sum(i * i for i in magnitudeSpectrum)

    rolloffIndex = 0
    rolloffSum = 0.0
    for i in range(0, len(magnitudeSpectrum)):
        rolloffSum += magnitudeSpectrum[i] * magnitudeSpectrum[i]
        if rolloffSum >= thresholdSum:
            rolloffIndex = i
            break

    return rolloffIndex / len(magnitudeSpectrum) * (fs / 2)


# Calculates the spectral centroid
def spectralCentroid(frame, previousFrame,  fs: int):
    magnitudeSpectrum = getMagnitudeSpectrum(frame)
    frameSize = len(frame)
    frequencies = np.abs(np.fft.fftfreq(frameSize, 1.0/fs)[:frameSize // 2 + 1])  # positive frequencies
    return sum(frequencies * magnitudeSpectrum) / sum(magnitudeSpectrum)


# x:            Audiosamples (list like)
# fs:           Sample rate
# frameSize:    Frame size in sample
# hopSize:      Hop size in sample
# filename:     Name of the file
def blockwiseFeature(x, fs: int, frameSize: int, hopSize: int, filename: str):
    rolloff = blockwise(x, frameSize, hopSize, spectralRolloff, [fs])
    centroid = blockwise(x, frameSize, hopSize, spectralCentroid, [fs])

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
    plt.show()


def main():
    # generate a 1 sec, 200 Hz triangle wave
    fs = 44100
    lengthInMs = 1000
    N = msToSamples(lengthInMs, fs)
    f0 = 200
    t = np.linspace(0, N, N)
    triangle = signal.sawtooth(2 * np.pi * f0 * t / fs, 0.5)

    fs, flute = getNormalizedAudio("flute_1.wav")

    frameSize = 2048
    hopSize = frameSize // 2
    blockwiseFeature(flute, fs, frameSize, hopSize, "flute_1.wav")

    ## Aufgabe 1: get_f0
    #print("Aufgabe 1 - Test get_f0: Grundfrequenz " + str(round(get_f0(triangle, fs), 3)) + " Hz")

    # ## Aufgabe 2: blockwise_f0
    # fs, violin = getNormalizedAudio("Violin_2.wav")
    #
    # frameSize = 2048
    # hopSize = frameSize // 2
    # f0s = blockwise_f0(violin, fs, frameSize, hopSize)
    # plt.plot(f0s)
    # plt.title('Aufgabe 2: Grundfrequenzverlauf der Datei "Violin_2.wav"')
    # plt.xlabel("Frame @" + str(frameSize) + " Samples (Hopsize: " + str(hopSize) + " Samples)")
    # plt.ylabel("Frequenz (in Hz)")
    # plt.show()
    #
    # ## Aufgabe 3: Testing
    #
    # # Triangle
    # frameSize = 1024
    # hopSize = frameSize // 2
    # triangle_f0s = blockwise_f0(triangle, fs, frameSize, hopSize)
    # plt.plot(triangle_f0s)
    # plt.title("Aufgabe 3 - Test: Triangle @" + str(f0) + " Hz")
    # plt.xlabel("Frame @" + str(frameSize) + " Samples (Hopsize: " + str(hopSize) + " Samples)")
    # plt.ylabel("Frequenz in Hz")
    # plt.ylim(f0 - 100, f0 + 100)
    # plt.show()
    #
    # # Sinus
    # f0 = 400
    # frameSize = 1024
    # hopSize = frameSize // 2
    # sinus = [math.sin(2 * math.pi * f0 * n / fs) for n in range(N)]
    # sinus_f0s = blockwise_f0(sinus, fs, frameSize, hopSize)
    # plt.plot(sinus_f0s)
    # plt.title("Aufgabe 3 - Test: Sinus @" + str(f0) + " Hz")
    # plt.xlabel("Frame @" + str(frameSize) + " Samples (Hopsize: " + str(hopSize) + " Samples)")
    # plt.ylabel("Frequenz in Hz")
    # plt.ylim(f0 - 100, f0 + 100)
    # plt.show()


#########################################
### Helper functions
#########################################


def count(iter):
    return sum(1 for _ in iter)


def getNormalizedAudio(filename: str):
    # read in audio and convert it to normalized floats
    fs, audio = scipy.io.wavfile.read(filename)
    maxVal = np.iinfo(audio.dtype).max
    audio = np.fromiter((s / maxVal for s in audio), dtype=float)
    return fs, audio


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
def blockwise(x, frameSize: int, hopSize: int, blockwiseFunction, functionArgs, zeroPad: bool = True):

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
