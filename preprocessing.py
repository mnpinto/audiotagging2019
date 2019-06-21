from fastai.vision import *
import librosa, librosa.display

# Based on: https://www.kaggle.com/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai

conf = {'sr': 44100, 'duration': 2, 'fmin': 20, 'n_mels': 128}
conf['hop_length'] = 347*conf['duration']
conf['fmax'], conf['n_fft'] = conf['sr'] // 2, conf['n_mels'] * 20
conf['samples'] = conf['sr'] * conf['duration']

def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf['sr'])
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf['samples']: # long enough
        if trim_long_data:
            y = y[0:0+conf['samples']]
    else: # pad blank
        padding = conf['samples'] - len(y) # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf['samples'] - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    conf2 = conf.copy()
    del conf2['duration'], conf2['samples']
    spectrogram = librosa.feature.melspectrogram(audio, **conf2)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', sr=conf['sr'], hop_length=conf['hop_length'], fmin=conf['fmin'], fmax=conf['fmax'])
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf['sr']))
        show_melspectrogram(conf, mels)
    return mels

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    Xstd = (X - mean) / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def convert_wav_to_image(df, fold, source):
    X = []
    for i, row in progress_bar(df.iterrows(), total=len(df)):
        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=False)
        x_color = mono_to_color(x)
        X.append(x_color)
    return X

def convert_wav_to_png(row):
    fname = row[1][0]
    x = read_as_melspectrogram(conf, path_source/str(fname), trim_long_data=False)
    x_color = mono_to_color(x)
    fsave = path_save/f'{fname[:-4]}.png'
    PIL.Image.fromarray(x_color).save(fsave)