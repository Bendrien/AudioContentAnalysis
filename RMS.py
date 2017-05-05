import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math


def getRMS(x):
    return math.sqrt(sum( i*i for i in x) / x.size);


def main():
    # read in audio and convert it to normalized floats
    fs, audio = scipy.io.wavfile.read("sinus_440Hz.wav");
    audio = np.fromiter((s / 32767 for s in audio), dtype=float);
    #print(audio);
    rms = getRMS(audio);
    print("RMS: " + str(rms));


if(__name__ == '__main__'):
    main()
