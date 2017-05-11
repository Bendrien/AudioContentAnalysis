import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math


# Correlates a signal with itself and returns the part from 0 <= i < ∞
def autocorrelate(x):
    result = np.correlate(x, x, mode='full')
    result = result[math.floor(len(result)/2):]
    # normalize
    maxVal = max(result)
    result = [s / maxVal for s in result]
    return result


# get the fundamental frequency of a signal
def get_f0(x, fs: int):
    correlation = autocorrelate(x)

    plt.plot(correlation)
    plt.show()

    # find the first peak right of the first zero crossing
    zeroCrossings = np.where(np.diff(np.signbit(correlation)))[0]
    firstZeroCrossingIndex = zeroCrossings[0]
    slicedCorrelation = correlation[firstZeroCrossingIndex:]
    maxValue = max(slicedCorrelation)
    maxIndex = slicedCorrelation.index(maxValue)
    peakIndex = maxIndex + firstZeroCrossingIndex

    f0 = fs / peakIndex

    # dumb check for validity
    if f0 > fs / 2:
        return 0

    return f0


def blockwise(x, fs: int, frameSizeInMS: int, hopSizeInMS: int):
    hopSize = msToSamples(hopSizeInMS, fs)  # in samples
    frameSize = msToSamples(frameSizeInMS, fs)  # in samples

    # zero pad at the end
    rest = len(x) % frameSize
    remainder = 0
    if rest != 0:
        remainder = frameSize - rest
    # print(remainder);
    x = np.lib.pad(x, (0, remainder), 'constant', constant_values=0)

    print(len(x))

    numberOfFrames = int(math.ceil(len(x) / hopSize))

    output = []
    for frameCount in range(0, numberOfFrames):
        begin = int(frameCount * hopSize)
        frame = x[begin: begin + frameSize]
        # print(len(frame));
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
    fs, audio = getNormalizedAudio("sawtooth.wav")

    f0 = get_f0(audio, fs)
    print("Grundfrequenz: " + str(round(f0, 3)) + " Hz")

    # Aufgabe 2: Blockwise RMS
    # fs, audio = getNormalizedAudio("git.wav")
    #
    # rms = blockwiseRMS(audio, fs, 20, 10, True)
    # rmsLog = blockwiseRMS(audio, fs, 20, 10, False)
    #
    # f, subplots = plt.subplots(2)
    # subplots[0].plot(rms)
    # subplots[0].set_title('RMS des git.wav')
    # subplots[0].set_ylabel("RMS (linear)")
    # subplots[1].plot(rmsLog)
    # subplots[1].set_xlabel("Time in Samples")
    # subplots[1].set_ylabel("RMS (dBFS)")
    # plt.show()


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
