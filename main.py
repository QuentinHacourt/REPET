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

    # power spectrogram is just the element-wise square of V
    V2 = V**2

    B = autocorrelation(V2)

    b = self_similarity(B)


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


def autocorrelation(matrix):
    num_bins, num_frames = matrix.shape
    B = np.zeros(matrix.shape)
    m = num_frames

    for i in range(num_bins):
        for j in range(num_frames):
            overlap_count = m - j
            sum_val = np.sum(matrix[i, :overlap_count] * matrix[i, j:])
            B[i, j] = sum_val / overlap_count

    return B


def self_similarity(matrix):
    n, m = matrix.shape
    b = np.zeros(m)

    for j in range(m):
        sum_val = 0
        for i in range(n):
            sum_val += matrix[i, j]

        b[j] = sum_val / n


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
