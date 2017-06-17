import numpy as np
import wavio
import matplotlib.pyplot as plt
import math
import nimfa


def main():
    frame = 512
    hop = frame//2

    fs, audio = getNormalizedAudio("train.wav")

    # calculate spectrum and initialize NMF
    V = np.array(blockwise(audio, frame, hop, getMagnitudeSpectrum))
    V = V.transpose()
    lsnmf = nimfa.Lsnmf(V, max_iter=20, rank=3)

    for i in range(20):

        lsnmf_fit = lsnmf.factorize()
        print(lsnmf_fit.fit.n_iter)
        W = lsnmf_fit.basis()
        H = lsnmf_fit.coef()

        f, axarr = plt.subplots(2, 2)
        plt.subplots_adjust(hspace=0.5)

        axarr[0, 0].matshow(V, aspect='auto', origin='lower')
        axarr[0, 0].set_title('Original spectrum')
        axarr[0, 0].xaxis.set_ticks_position('bottom')

        axarr[0, 1].matshow(W, aspect='auto', origin='lower')
        axarr[0, 1].set_title('W')
        axarr[0, 1].xaxis.set_ticks_position('bottom')

        axarr[1, 0].matshow(H, aspect='auto', origin='lower')
        axarr[1, 0].set_title('H')
        axarr[1, 0].xaxis.set_ticks_position('bottom')

        axarr[1, 1].matshow(W*H, aspect='auto', origin='lower')
        axarr[1, 1].set_title('W*H')
        axarr[1, 1].xaxis.set_ticks_position('bottom')

        plt.show()


#########################################
### Helper functions
#########################################


def getMagnitudeSpectrum(frame, _):
    return abs(np.fft.rfft(frame))


def getNormalizedAudio(filename: str):
    # read in audio and convert it to normalized floats
    wav = wavio.read(filename)
    audio = wav.data
    fs = wav.rate
    maxVal = np.iinfo(audio.dtype).max
    audio = np.fromiter((s / maxVal for s in audio), dtype=float)
    return fs, audio


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
