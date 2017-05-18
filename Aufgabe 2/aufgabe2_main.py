import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math
import itertools


def count(iter):
    return sum(1 for _ in iter)

# Correlates a signal with itself and returns the part from 0 <= i < ∞
def xcorr(x):
    result = np.correlate(x, x, mode='full')
    result = result[math.floor(len(result)/2):]
    # normalize
    maxVal = max(result)
    result = [s / maxVal for s in result]
    return result


# get the fundamental frequency of a signal
def get_f0(x, fs: int):
    x_rr = xcorr(x)

    # index after first zero crossing
    i0 = count(itertools.takewhile(lambda x: x >= 0, x_rr))
    slice = x_rr[i0:]
    # index of the second peak
    peakIndex = i0 + slice.index(max(slice))
    if peakIndex == 0:
        return 0

    f0 = fs / peakIndex

    # dumb check for validity
    if f0 > fs / 2:
        return 0

    return f0


# x:         Audio samples (list like)
# fs:        Sample Rate
# frameSize: Frame size in sample
# hopSize:   Hop size in sample
def blockwise_f0(x, fs: int, frameSize: int, hopSize: int):
    # zero pad at the end
    rest = len(x) % hopSize
    if rest != 0:
        remainder = frameSize - rest
    else:
        remainder = hopSize
    x = np.lib.pad(x, (0, remainder), 'constant', constant_values=0)

    # calculate the number of blocks
    numberOfFrames = int(math.ceil(len(x) / hopSize) - 1)  # minus 1, because we zero padded
    output = []

    # blockwise f0
    for frameCount in range(0, numberOfFrames):
        begin = int(frameCount * hopSize)
        frame = x[begin: begin + frameSize]
        output.append(get_f0(frame, fs))

    return output


def getNormalizedAudio(filename: str):
    # read in audio and convert it to normalized floats
    fs, audio = scipy.io.wavfile.read(filename)
    maxVal = np.iinfo(audio.dtype).max
    audio = np.fromiter((s / maxVal for s in audio), dtype=float)
    return fs, audio


def msToSamples(timeInMs: int, fs: int):
    return int(math.ceil(timeInMs / 1000 * fs))


def main():
    # Aufgabe 1: get_f0
    fs, saw = getNormalizedAudio("sawtooth.wav")

    f0 = get_f0(saw, fs)
    print("Grundfrequenz: " + str(round(f0, 3)) + " Hz")

    # Aufgabe 2: blockwise_f0
    fs, violin = getNormalizedAudio("Violin_2.wav")

    frameSize = 2048
    hopSize = int(frameSize / 2)
    f0s = blockwise_f0(violin, fs, frameSize, hopSize)
    plt.plot(f0s)
    # plt.ylim(-1, 1)
    plt.ylabel("Frequenz (in Hz)")
    plt.title('Grundfrequenzverlauf der Datei "Violin_2.wav"')
    plt.show()

    # Aufgabe 3: Testing
    # fs = 44100
    # lengthInMs = 100
    # f0 = 999
    # N = msToSamples(lengthInMs, fs)
    # dcOffsetSignal   = [0.5 for n in range(N)]
    # nullSignal       = [0 for n in range(N)]
    # sinus999HzSignal = [math.sin(2 * math.pi * f0 * n / fs) for n in range(N)]
    #
    # dcRMS = blockwiseRMS(dcOffsetSignal, fs, 20, 10, False)
    # print(dcRMS)
    # plt.plot(dcRMS)
    # plt.ylim(-1, 1)
    # plt.ylabel("RMS (linear)")
    # plt.title("0.5 DC Offset")
    # plt.show()
    #
    # nullRMS = blockwiseRMS(nullSignal, fs, 20, 10, False)
    # plt.plot(nullRMS)
    # plt.ylim(-1, 1)
    # plt.ylabel("RMS (linear)")
    # plt.title("Nullvektor")
    # plt.show()
    # # padded signal = 4851
    #
    # sinusRMS = blockwiseRMS(sinus999HzSignal, fs, 20, 10, False)
    # plt.plot(sinusRMS)
    # plt.ylim(-1, 1)
    # plt.ylabel("RMS (linear)")
    # plt.title("Sinus 999Hz")
    # plt.show()

if __name__ == '__main__':
    main()
