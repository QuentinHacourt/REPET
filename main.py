import sys
import os
import librosa
import soundfile as sf
import numpy as np


def main():
    if len(sys.argv) < 2:
        print(
            "ERROR: Please enter a valid .wav file name for this program to run correctly"
        )
        return

    filename = sys.argv[1]
    repet(filename)


def repet(filename):
    # get signal x
    x, sr = load_file(filename)

    # calculate short time fourier transform
    X = stft(x)

    # get the magnitude spectrogram V
    V = np.abs(X)


def load_file(filename):
    if file_exists(filename):
        print(f"Loading: {filename}...")
        # load audio file
        # sr = sampling rate
        # sr=None preserves the sampling rate
        # mono=True converts signal to mono

        x, sr = librosa.load(filename, sr=None, mono=True)
        print("Loading done!")
        return x, sr
    else:
        print(f"ERROR: there is no file called: {filename}")
        sys.exit(1)


# takes an audio signal and returns the result of its short time fourier transform
def stft(audio_signal):
    n_fft = 1024
    hop_length = 512

    # Hamming window according to paper
    X = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length, window="hamming")

    return X


def file_exists(filename):
    if os.path.isfile(filename):
        return True

    return False


if __name__ == "__main__":
    main()
