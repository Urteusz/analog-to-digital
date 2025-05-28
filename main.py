import sounddevice as sd
import soundfile as sf
import numpy as np
import os


def list_input_devices():
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    for i, d in enumerate(input_devices):
        print(f"{i}: {d['name']} (ID: {devices.index(d)})")
    return input_devices


def record_audio(filename, samplerate, duration, bit_depth, device):
    print(f"\nNagrywanie {duration} sekund dźwięku ({samplerate} Hz, {bit_depth}-bit, urządzenie ID: {device})...")
    try:
        audio_data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            device=device
        )
        sd.wait()
    except Exception as e:
        print(f"Błąd nagrywania: {e}")
        return False

    max_val = 2 ** (bit_depth - 1) - 1
    quantized_audio = np.int16(audio_data[:, 0] * max_val)

    try:
        sf.write(filename, quantized_audio, samplerate, subtype=f'PCM_{bit_depth}')
        print(f"Zapisano plik: {filename}")
        return True
    except Exception as e:
        print(f"Błąd zapisu: {e}")
        return False


def play_audio(filename):
    try:
        data, samplerate = sf.read(filename, dtype='float32')
        if data.size == 0:
            print(f"Plik {filename} jest pusty.")
            return
        print(f"Odtwarzanie: {filename}")
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"Błąd odtwarzania: {e}")


def main():
    print("Wybierz urządzenie wejściowe (mikrofon):")
    input_devices = list_input_devices()
    dev_index = int(input("Twój wybór: "))
    device_id = sd.query_devices().index(input_devices[dev_index])

    samplerates = [8000, 16000, 44100, 48000, 96000]
    bit_depths = [8, 16, 24]

    print("\nWybierz częstotliwość próbkowania:")
    for i, sr in enumerate(samplerates):
        print(f"{i}: {sr} Hz")
    sr_index = int(input("Twój wybór: "))
    samplerate = samplerates[sr_index]

    print("\nWybierz głębokość bitową:")
    for i, bd in enumerate(bit_depths):
        print(f"{i}: {bd} bit")
    bd_index = int(input("Twój wybór: "))
    bit_depth = bit_depths[bd_index]

    try:
        duration = float(input("\nCzas nagrania (w sekundach): "))
    except ValueError:
        print("Nieprawidłowy czas. Używam domyślnych 5 sekund.")
        duration = 5

    filename = f"nagranie_{samplerate}Hz_{bit_depth}bit.wav"
    os.makedirs("wyniki", exist_ok=True)
    filepath = os.path.join("wyniki", filename)

    if record_audio(filepath, samplerate, duration, bit_depth, device_id):
        play_audio(filepath)


if __name__ == "__main__":
    main()
