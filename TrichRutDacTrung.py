from pydub import AudioSegment
import numpy as np
import array
import struct
import math

'''
    Trích rút đặc trưng
'''


def features(file):
    audio_file = readFileAudio(file)
    frames = audioFraming(audio_file)
    features = []
    for frame in frames:
        if (checkSilence(frame)):
            z = zeroCrossingRate(frame)
            ae = averageEnergy(frame)
            af = averageFrequency(frame)
            vf = frequencyVariation(frame)
            ap = averagePitch(frame)
            vp = pitchVariation(frame)
            features.append([ae, z, af, vf, ap, vp])
    return features


'''
    Sử dụng AudioSegment để đọc file âm thanh từ đường dẫn
'''


def readFileAudio(fileName):
    # Load audio file
    audio_file = AudioSegment.from_wav(fileName)

    # Convert to mono channel
    audio_file = audio_file.set_channels(1)

    return audio_file


'''
    Chia file âm thanh thành các đoạn có độ dài 0.5s, mỗi đoạn cách nhau 0.25s
    VD: file âm thanh 10s -> Chia được thành (10/0.25)-1=39 frame
'''


def audioFraming(audio_file):
    # Set frame length and hop length in milliseconds
    frame_length = 500  # Do dai: 0.5s
    hop_length = 250  # Khoang cach giua cac frame: 0.25s

    # Calculate number of frames
    num_frames = int(len(audio_file) / hop_length) - 1

    # Initialize empty list for frames
    frames = []

    # Cut audio file into frames
    for i in range(num_frames):
        start_time = i * hop_length
        end_time = start_time + frame_length
        frame = audio_file[start_time:end_time]
        frames.append(frame)

    return frames


'''
    Kiểm tra đoạn âm thanh có phải im lặng ko
    Nếu >= 80% đoạn âm thanh im lặng thì đoạn âm thanh đó im lặng
'''


def checkSilence(audio):
    # Độ sâu bit (hay Bit depth) là số bit được sử dụng để biểu diễn độ lớn của tín hiệu âm thanh
    # Tệp âm thanh 16 bit (Kích thước mẫu) bits per sample (chỉ số bit được dùng để mã hóa 1 sample)
    # Tần số lấy mẫu frame_rate 44100 Hz
    # array_type: h signed short
    # print(array.array("h", audio._data))
    samples = np.array(audio.get_array_of_samples())
    # print(len(audio._data))
    # Lấy mảng các mẫu âm thanh (giá trị 1 mẫu nằm trong khoảng từ -32768 đến 32767)
    # Giá trị mẫu càng nhỏ thì âm thanh càng lớn

    # ngưỡng độ lớn tạm dừng (silence threshold)
    threshold = 280

    # Tính toán số mẫu (samples) trong file âm thanh có độ lớn nhỏ hơn ngưỡng độ lớn tạm dừng
    silence_samples = len(np.where(abs(samples) < threshold)[0])

    # Tính toán tỷ lệ câm (silence ratio) của file âm thanh
    silence_ratio = silence_samples / len(samples)
    if (silence_ratio >= 0.8):
        return False
    return True


'''
    Tính tỷ lệ qua điểm 0
'''


def zeroCrossingRate(audio):
    # Tỷ lệ qua điểm zero
    # Extract samples from audio file
    samples = np.array(audio.get_array_of_samples())
    # Calculate zero crossing rate
    count_zero = np.sum(np.abs(np.diff(np.sign(samples))))/2
    zero_crossing_rate = count_zero / len(samples)

    # Return zero crossing rate
    return zero_crossing_rate


'''
    Tính năng lượng trung bình
'''


def _get_samples(cp, size):
    for i in range(22050):
        yield _get_sample(cp, size, i)

# Lấy mẫu từ dữ liệu thô


def _get_sample(cp, size, i):
    global z
    start = i * size
    end = start + size
    return struct.unpack_from('h', cp[start:end])[0]


def rms(cp, size):
    sample_count = 22050
    sum_squares = sum(sample**2 for sample in _get_samples(cp, size))
    return int(math.sqrt(sum_squares / sample_count))


def averageEnergy(audio):
    # Calculate RMS energy
    # rms_energy = audio.rms
    rms_energy = rms(audio._data, 2)

    # Return RMS energy
    return rms_energy


'''
    Tính tần số trung bình (Spectral Centroid)
'''


def averageFrequency(audio):
    # Chuyển đổi tín hiệu âm thanh sang miền tần số
    samples = np.array(audio.get_array_of_samples())

    # Biến đổi fourier
    frequencies = np.fft.fftfreq(
        len(samples), d=1.0/audio.frame_rate)  # Tần số cơ bản
    spectral = np.fft.fft(samples)  # Mảng biên độ của các tần số
    # Tính trung bình các tần số
    avg_freq = np.abs(spectral).dot(np.abs(frequencies)) / \
        np.sum(np.abs(spectral))  # Tích vô hướng/Tổng biên độ

    return avg_freq


'''
    Tính độ biến thiên tần số
'''


def frequencyVariation(audio):
    # Chuyển đổi tín hiệu âm thanh sang miền thời gian
    samples = np.array(audio.get_array_of_samples())
    # frequencies = np.fft.fftfreq(len(samples), d=1.0/audio.frame_rate)
    spectral = np.fft.fft(samples)  # Phổ (biên độ tần số)

    # Tính độ biến thiên tần số
    # Tính hiệu biên độ tần số cạnh nhau
    diff_freq = np.abs(np.diff(spectral))
    mean_diff_freq = np.mean(diff_freq, axis=0)
    return mean_diff_freq


'''
    Tính độ cao trung bình
'''


def averagePitch(audio):
    # Chuyển đổi tín hiệu âm thanh sang mảng numpy
    samples = np.array(audio.get_array_of_samples())

    # Tính phổ tín hiệu âm thanh
    spectrum = np.fft.fft(samples)

    # Tính giá trị tần số cơ bản (fundamental frequency)
    # Tính các tần số tương ứng cho các thành phần của phổ tín hiệu
    freqs = np.fft.fftfreq(len(samples), d=1.0/audio.frame_rate)
    # d=1.0/audio.frame_rate là khoảng thời gian giữa các mẫu (nghịch đảo của tốc độ lấy mẫu)

    # Tìm các vị trí trong mảng freqs mà giá trị tần số tại đó lớn hơn 0
    pos_mask = np.where(freqs > 0)

    # Giữ lại các tần số dương trong mảng freqs
    freqs = freqs[pos_mask]

    # Lọc phổ tương ứng với các tần số dương
    spectrum = spectrum[pos_mask]

    # Tìm tần số có biên độ lớn nhất
    peak = np.argmax(np.abs(spectrum))
    # np.argmax(np.abs(spectrum)): Tìm vị trí của giá trị biên độ lớn nhất trong phổ tín hiệu
    # np.abs(spectrum) lấy giá trị tuyệt đối của phổ, vì phổ tín hiệu là số phức

    # Tần số cơ bản là tần số tại vị trí có biên độ lớn nhất
    fundamental_freq = freqs[peak]

    # Tính pitch từ giá trị tần số cơ bản
    pitch = 0.0
    if fundamental_freq > 0:
        # Nếu tần số cơ bản > 0, tính pitch bằng cách chia tần số lấy mẫu cho tần số cơ bản
        pitch = audio.frame_rate / fundamental_freq
    # Nếu tần số cơ bản <= 0, độ cao trung bình là 0.0
    return pitch


'''
    Tính độ biến thiên cao độ
'''


def pitchVariation(audio):

    # Chuyển đổi tín hiệu âm thanh sang mảng numpy (miền tần số)
    samples = np.array(audio.get_array_of_samples())

    # Tính giá trị tần số cơ bản (fundamental frequency) trong các khung thời gian nhỏ
    # Kích thước cửa sổ trượt (Tần số lấy mẫu / 100)
    window_size = int(audio.frame_rate / 100.0)

    # Khoảng cách giữa các cửa sổ (Khoảng chồng lấn giữa các cửa sổ)
    hop_size = window_size // 2

    fundamental_freqs = []

    # Duyệt từng cửa sổ có kích thước window_size
    for i in range(0, len(samples) - window_size, hop_size):
        window = samples[i:i+window_size]

        # Tính phổ tín hiệu (chuyển từ miền thời gian sang miền tần số)
        spectrum = np.fft.fft(window)

        # Tính tần số cơ bản
        freqs = np.fft.fftfreq(len(window), d=1.0/audio.frame_rate)

        # Tìm các vị trí trong mảng freqs mà giá trị tần số tại đó lớn hơn 0
        pos_mask = np.where(freqs > 0)

        # Tính phổ tín hiệu âm thanh
        spectrum = spectrum[pos_mask]

        # Giữ lại các tần số dương trong mảng freqs
        freqs = freqs[pos_mask]

        # Tìm tần số có biên độ lớn nhất
        peak = np.argmax(np.abs(spectrum))

        # Tần số cơ bản là tần số tại vị trí có biên độ lớn nhất
        fundamental_freq = freqs[peak]

        fundamental_freqs.append(fundamental_freq)

    # Tính độ biến thiên pitch
    # Tính hiệu tần số cơ bản cạnh nhau
    diff_freqs = np.abs(np.diff(fundamental_freqs))

    # Tính giá trị trung bình
    mean_diff_freqs = np.mean(diff_freqs)
    return mean_diff_freqs
