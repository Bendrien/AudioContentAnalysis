import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math
import itertools


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
    i0 = count(itertools.takewhile(lambda x: x >= 0, x_rr));
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
    # calculate the number of blocks
    numberOfFrames = (len(x) // hopSize) - 1
    output = []

    # blockwise f0
    for frameCount in range(0, numberOfFrames):
        begin = int(frameCount * hopSize)
        frame = x[begin: begin + frameSize]
        output.append(get_f0(frame, fs))

    return output


def main():
    ## Aufgabe 1: get_f0
    fs, saw = getNormalizedAudio("sawtooth.wav")

    f0 = get_f0(saw, fs)
    print("Grundfrequenz: " + str(round(f0, 3)) + " Hz")

    ## Aufgabe 2: blockwise_f0
    fs, violin = getNormalizedAudio("Violin_2.wav")

    frameSize = 2048
    hopSize = frameSize // 2
    f0s = blockwise_f0(violin, fs, frameSize, hopSize)
    plt.plot(f0s)
    plt.title('Aufgabe 2: Grundfrequenzverlauf der Datei "Violin_2.wav"')
    plt.xlabel("Frame @" + str(frameSize) + " Samples (Hopsize: " + str(hopSize) + " Samples)")
    plt.ylabel("Frequenz (in Hz)")
    plt.show()

    ## Aufgabe 3: Testing
    fs = 44100
    lengthInMs = 1000
    N = msToSamples(lengthInMs, fs)

    # Sinus
    f0 = 400
    frameSize = 1024
    hopSize = frameSize // 2
    sinus = [math.sin(2 * math.pi * f0 * n / fs) for n in range(N)]
    sinus_f0s = blockwise_f0(sinus, fs, frameSize, hopSize)
    plt.plot(sinus_f0s)
    plt.title("Aufgabe 3 - erster Test: Sinus @" + str(f0) + " Hz")
    plt.xlabel("Frame @" + str(frameSize) + " Samples (Hopsize: " + str(hopSize) + " Samples)")
    plt.ylabel("Frequenz in Hz")
    plt.ylim(300, 500)
    plt.show()

    # Saw
    frameSize = 512
    hopSize = frameSize // 2
    fs, saw = getNormalizedAudio("sawtooth.wav")
    saw_f0s = blockwise_f0(saw, fs, frameSize, hopSize)
    plt.plot(saw_f0s)
    plt.title("Aufgabe 3 - zweiter Test: Sawtooth @200 Hz")
    plt.xlabel("Frame @" + str(frameSize) + " Samples (Hopsize: " + str(hopSize) + " Samples)")
    plt.ylabel("Frequenz in Hz")
    plt.ylim(0, 300)
    plt.show()


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


if __name__ == '__main__':
    main()
