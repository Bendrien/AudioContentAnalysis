import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math


def getRMS(x, isLog):
    rms = math.sqrt(sum(i*i for i in x) / len(x));

    if(isLog):
        if(rms == 0):
            return -math.inf;
        rms = 20 * math.log10(math.fabs(rms));

    return rms;

def blockwiseRMS(x, fs, frameSizeInMS, hopSizeInMS, isLog):
    hopSize = math.ceil(hopSizeInMS / 1000 * fs); # in samples
    frameSize = math.ceil(frameSizeInMS / 1000 * fs); # in samples
    #x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    #hopSize = 3;
    #frameSize = 5;

    #print(x);

    # zero pad at the end
    remainder = frameSize - (len(x) % frameSize);
    #print(remainder);
    x = np.lib.pad(x, (0, remainder), 'constant', constant_values=(0));

    #print(x);

    numberOfFrames = math.ceil((len(x) / hopSize) - 1);

    #print(numberOfFrames);

    output = [];
    for frameCount in range(0, numberOfFrames):
        begin = int(frameCount * hopSize);
        frame = x[begin : begin + frameSize];
        #print(len(frame));
        output.append(getRMS(frame, isLog));

    return output;

def getNormalizedAudio(filename):
    # read in audio and convert it to normalized floats
    fs, audio = scipy.io.wavfile.read(filename);
    maxVal = np.iinfo(audio.dtype).max;
    audio = np.fromiter((s / maxVal for s in audio), dtype=float);
    return fs, audio;

def main():
    # Aufgabe 1: RMS
    fs, audio = getNormalizedAudio("sinus_440Hz.wav");

    rms = getRMS(audio, False);
    print("RMS: " + str(rms));

    # Aufgabe 2: Blockwise RMS
    fs, audio = getNormalizedAudio("git.wav");

    rms = blockwiseRMS(audio, fs, 20, 10, True);
    plt.plot(rms);
    plt.show();

    rms = blockwiseRMS(audio, fs, 20, 10, False);
    plt.plot(rms);
    plt.show();

    # Aufgabe 3: Testing
    fs = 44100;
    lengthInMs = 100;
    f0 = 999;
    N = math.ceil(lengthInMs / 1000 * fs); # in samples
    dcOffsetSignal   = [0.5 for n in range(N)];
    nullSignal       = [0 for n in range(N)];
    sinus999HzSignal = [math.sin(2 * math.pi * f0 * n / fs) for n in range(N)];

    dcRMS = blockwiseRMS(dcOffsetSignal, fs, 20, 10, False);
    print(dcRMS);
    plt.plot(dcRMS);
    plt.ylim(-1, 1);
    plt.title("0.5 DC Offset");
    plt.show();

    nullRMS = blockwiseRMS(nullSignal, fs, 20, 10, True);
    plt.plot(nullRMS);
    plt.ylim(-1, 1);
    plt.title("Nullvektor");
    plt.show();

    sinusRMS = blockwiseRMS(sinus999HzSignal, fs, 20, 10, False);
    plt.plot(sinusRMS);
    plt.ylim(-1, 1);
    plt.show();

if(__name__ == '__main__'):
    main()
