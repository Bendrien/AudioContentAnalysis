import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math


def getRMS(x, isLog: bool):
    # calculate RMS
    rms = math.sqrt(sum(i*i for i in x) / len(x))

    # scale to logarithmic dBFS if wanted
    if isLog:
        if rms == 0:
            return -math.inf
        rms = 20 * math.log10(math.fabs(rms))

    return rms


def blockwiseRMS(x, fs: int, frameSizeInMS: int, hopSizeInMS: int, isLog: bool):
    hopSize = msToSamples(hopSizeInMS, fs)  # in samples
    frameSize = msToSamples(frameSizeInMS, fs)  # in samples

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

    # blockwise RMS
    for frameCount in range(0, numberOfFrames):
        begin = int(frameCount * hopSize)
        frame = x[begin: begin + frameSize]
        output.append(getRMS(frame, isLog))

    return output


# read in audio and convert it to normalized floats
def getNormalizedAudio(filename: str):
    fs, audio = scipy.io.wavfile.read(filename)
    maxVal = np.iinfo(audio.dtype).max
    audio = np.fromiter((s / maxVal for s in audio), dtype=float)
    return fs, audio


# convert milliseconds to samples
def msToSamples(timeInMs: int, fs: int):
    return int(math.ceil(timeInMs / 1000 * fs))


def main():
    # Aufgabe 1: RMS
    fs, audio = getNormalizedAudio("sinus_440Hz.wav")

    rms = getRMS(audio, False)
    print("RMS: " + str(rms))

    # Aufgabe 2: Blockwise RMS
    fs, audio = getNormalizedAudio("git.wav")

    rms = blockwiseRMS(audio, fs, 20, 10, False)
    rmsLog = blockwiseRMS(audio, fs, 20, 10, True)

    f, subplots = plt.subplots(2)
    subplots[0].plot(rms)
    subplots[0].set_title('RMS des git.wav')
    subplots[0].set_ylabel("RMS (linear)")
    subplots[1].plot(rmsLog)
    subplots[1].set_xlabel("Time in Samples")
    subplots[1].set_ylabel("RMS (dBFS)")
    plt.show()


    # Aufgabe 3: Testing
    fs = 44100
    lengthInMs = 100
    f0 = 999
    N = msToSamples(lengthInMs, fs)
    dcOffsetSignal   = [0.5 for n in range(N)]
    nullSignal       = [0 for n in range(N)]
    sinus999HzSignal = [math.sin(2 * math.pi * f0 * n / fs) for n in range(N)]

    dcRMS = blockwiseRMS(dcOffsetSignal, fs, 20, 10, False)
    plt.plot(dcRMS)
    plt.ylim(-1, 1)
    plt.ylabel("RMS (linear)")
    plt.title("0.5 DC Offset")
    plt.show()

    nullRMS = blockwiseRMS(nullSignal, fs, 20, 10, False)
    plt.plot(nullRMS)
    plt.ylim(-1, 1)
    plt.ylabel("RMS (linear)")
    plt.title("Nullvektor")
    plt.show()
    # padded signal = 4851

    sinusRMS = blockwiseRMS(sinus999HzSignal, fs, 20, 10, False)
    plt.plot(sinusRMS)
    plt.ylim(-1, 1)
    plt.ylabel("RMS (linear)")
    plt.title("Sinus 999Hz")
    plt.show()


if __name__ == '__main__':
    main()
