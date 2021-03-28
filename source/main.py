from scipy import signal
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment

output_path = 'audio/temp.wav'


def convertstereotomono(stereo_wav_path):
    sound = AudioSegment.from_wav(stereo_wav_path)
    sound = sound.set_channels(1)
    sound.export(output_path, format="wav")


def main():
    # samplerate, data = wavfile.read('C:/Users/smithdepazd/Projects/silbo_gomero_translator/source/audio/500_4000_1300_500.wav')
    file_path = 'audio/buenos_dias1.wav'

    convertstereotomono(file_path)

    sample_rate, samples = wavfile.read(output_path)

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, scaling='spectrum')

    plt.pcolormesh(times, frequencies, spectrogram)

    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.show()
    print('hello')


def nonsense():
    import numpy as np
    from matplotlib import pyplot as plt
    import scipy.io.wavfile as wav
    from numpy.lib import stride_tricks

    """ short time fourier transform of audio signal """

    def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)
        samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
        # cols for windowing
        cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(frameSize))

        frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                          strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
        frames *= win

        return np.fft.rfft(frames)

    """ scale frequency axis logarithmically """

    def logscale_spec(spec, sr=44100, factor=20.):
        timebins, freqbins = np.shape(spec)

        scale = np.linspace(0, 1, freqbins) ** factor
        scale *= (freqbins - 1) / max(scale)
        scale = np.unique(np.round(scale))

        # create spectrogram with new freq bins
        newspec = np.complex128(np.zeros([timebins, len(scale)]))
        for i in range(0, len(scale)):
            if i == len(scale) - 1:
                newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
            else:
                newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

        # list center freq of bins
        allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
        freqs = []
        for i in range(0, len(scale)):
            if i == len(scale) - 1:
                freqs += [np.mean(allfreqs[int(scale[i]):])]
            else:
                freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

        return newspec, freqs

    """ plot spectrogram"""

    # nipy_spectral
    #
    def plotstft(audiopath, binsize=2 ** 12, plotpath=None, colormap="nipy_spectral"):
        samplerate, samples = wav.read(audiopath)

        s = stft(samples, binsize)

        sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
        # arr = np.array(freq)
        # filter_f = 5000 > arr.all() > 500
        #
        # freq = arr[filter_f]

        ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
        # for x,f in enumerate(ims):
        #   filter_f = f < 200
        #   ims[x] = f[filter_f].resize(ims[x].size)

        # ims = ims[filter_ims]
        # for x,i in enumerate(ims):
        #   ims[x] = np.resize(i,(i.size)//2)

        # for j in i:
        #   print(j)

        modified = synthesize_data(ims)
        timebins, freqbins = np.shape(modified)

        print("timebins: ", timebins)
        print("freqbins: ", freqbins)

        plt.figure(figsize=(15, 7.5))
        plt.imshow(np.transpose(modified), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        plt.colorbar()

        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins - 1])
        plt.ylim([0, freqbins])

        xlocs = np.float32(np.linspace(0, timebins - 1, 5))
        plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate])
        ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
        plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

        if plotpath:
            plt.savefig(plotpath, bbox_inches="tight")
        else:
            plt.show()

        plt.clf()

        return ims

    ims = plotstft('audio/buenos_dias1.wav')


def synthesize_data(data):
    output = []

    for i in data:
        # filt = i > 200
        # i[i < 200] = 0
        i = i[0:len(i) // 7]

        i = np.where(i < 220, 0, i)

        # skip_flag = True
        # for n in i:
        #   if n > 1 :
        #     skip_flag = False
        #     break
        #
        # if skip_flag:
        #   pass
        # else:
        output.append(i)
    return output

# ¯	= high vowel (e,i)
# _ = low vowel (a,o,u)

# Y = high-continuous consonant (n,ñ,l,ll,y,rr,r,d)
# T = high-interrupted consonant (t,ch,s)

# G = low-continuous consonant (m,b,f,g,h)
# K = low-interrupted consonant (p,k)


if __name__ == '__main__':
    # main()
    nonsense()
