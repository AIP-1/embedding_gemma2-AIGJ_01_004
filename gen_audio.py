import subprocess
import numpy as np
import soundfile as sf
import librosa

SR = 16000
rng = np.random.default_rng(0)

subprocess.run(["espeak-ng", "-v", "ko", "강의실이 너무 덥습니다. 에어컨을 켜 주세요.",
                "-w", "media/raw_speech.wav"], check=True)
y, _ = librosa.load("media/raw_speech.wav", sr=SR, mono=True)
sf.write("media/speech.wav", y, SR)

t = np.arange(3 * SR) / SR
sf.write("media/beep.wav", 0.5 * np.sin(2 * np.pi * 1000 * t), SR)

white = np.fft.rfft(rng.standard_normal(5 * SR))
f = np.fft.rfftfreq(5 * SR, 1 / SR)
f[0] = f[1]
pink = np.fft.irfft(white / np.sqrt(f), n=5 * SR)
sf.write("media/noise.wav", 0.5 * pink / np.abs(pink).max(), SR)

for name in ["speech", "beep", "noise"]:
    info = sf.info(f"media/{name}.wav")
    print(f"{name}.wav: {info.samplerate} Hz, {info.channels}채널, {info.duration:.1f}초")
