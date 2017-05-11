import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math


def getRMS(x):
    return math.sqrt(sum( i*i for i in x) / len(x));

def blockwiseRMS(x, fs, frameSizeInMS, hopSizeInMS):
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
        print(len(frame));
        output.append(getRMS(frame));

    return output;

def getNormalizedAudio(filename):
    # read in audio and convert it to normalized floats
    fs, audio = scipy.io.wavfile.read(filename);
    maxVal = np.iinfo(audio.dtype).max;
    audio = np.fromiter((s / maxVal for s in audio), dtype=float);
    return fs, audio;

def main():
    fs, audio = getNormalizedAudio("sinus_440Hz.wav");

    rms = getRMS(audio);
    print("RMS: " + str(rms));


    fs, audio = getNormalizedAudio("git.wav");

    rms = blockwiseRMS(audio, fs, 20, 10);
    plt.plot(rms);
    plt.show();


if(__name__ == '__main__'):
    main()
