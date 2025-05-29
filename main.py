import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import scipy.signal


def list_input_devices():
    """Wyświetla dostępne urządzenia wejściowe audio"""
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    print("Dostępne urządzenia wejściowe:")
    for i, d in enumerate(input_devices):
        print(f"{i}: {d['name']} (ID: {devices.index(d)})")
    return input_devices


def record_audio(samplerate, duration, device):
    """Nagrywa dźwięk z wybranego urządzenia"""
    print(f"\nNagrywanie {duration} sekund dźwięku ({samplerate} Hz, urządzenie ID: {device})...")
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                       channels=1, dtype='float32', device=device)
        sd.wait()
        print("Nagrywanie zakończone.")
        return audio[:, 0]  # zwracamy 1D tablicę
    except Exception as e:
        print(f"Błąd nagrywania: {e}")
        return None


def quantize(audio, bit_depth):
    """Kwantyzuje sygnał audio do określonej głębokości bitowej"""
    if bit_depth == 32:
        return audio  # Brak kwantyzacji dla 32-bit float

    max_val = 2 ** (bit_depth - 1) - 1
    quantized = np.round(np.clip(audio, -1, 1) * max_val) / max_val
    return quantized


def resample_audio(audio, orig_sr, target_sr):
    """Zmienia częstotliwość próbkowania sygnału"""
    if orig_sr == target_sr:
        return audio
    return scipy.signal.resample_poly(audio, target_sr, orig_sr)


def save_audio(filename, audio, samplerate, bit_depth):
    """Zapisuje sygnał audio do pliku WAV z określonymi parametrami"""
    try:
        if bit_depth == 8:
            # 8-bit unsigned: przeskaluj z (-1.0, 1.0) na (0.0, 1.0)
            audio_scaled = np.clip((audio + 1.0) / 2.0, 0.0, 1.0).astype('float32')
            sf.write(filename, audio_scaled, samplerate, subtype='PCM_U8')
        elif bit_depth == 12:
            # Symulacja 12-bit przez zapisanie jako 16-bit z ograniczoną precyzją
            audio_12bit = quantize(audio, 12)
            sf.write(filename, (audio_12bit * 32767).astype('int16'), samplerate, subtype='PCM_16')
        elif bit_depth == 16:
            sf.write(filename, (audio * 32767).astype('int16'), samplerate, subtype='PCM_16')
        elif bit_depth == 24:
            sf.write(filename, (audio * (2 ** 23 - 1)).astype('int32'), samplerate, subtype='PCM_24')
        elif bit_depth == 32:
            sf.write(filename, audio.astype('float32'), samplerate, subtype='FLOAT')
        else:
            raise ValueError(f"Nieobsługiwana głębokość bitowa: {bit_depth}")

        print(f"Zapisano: {filename}")
        return True
    except Exception as e:
        print(f"Błąd zapisu {filename}: {e}")
        return False


def compute_snr(original, distorted):
    """Oblicza stosunek sygnał/szum (SNR) w dB"""
    noise = original - distorted
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def save_results_report(results):

    with open("wyniki/sprawozdanie.txt", "w", encoding="utf-8") as f:
        f.write("SPRAWOZDANIE Z PRZETWARZANIA ANALOGOWO-CYFROWEGO (A/C)\n")

        f.write("WYNIKI ANALIZY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Częst.[Hz]':<10} {'Bit':<4} {'SNR[dB]':<8}\n")
        f.write("-" * 40 + "\n")

        for result in sorted(results, key=lambda x: (x['snr']), reverse=True):
            f.write(f"{result['samplerate']:<10} {result['bit_depth']:<4} "
                    f"{result['snr']:<8.2f} \n")

        print(f"Sprawozdanie zapisane do: wyniki/sprawozdanie.txt")


def main():
    print("PRZETWORNIK ANALOGOWO-CYFROWY (A/C) I CYFROWO-ANALOGOWY (C/A)")
    print("=" * 65)

    # Tworzenie katalogu wyników
    os.makedirs("wyniki", exist_ok=True)

    # Wybór urządzenia wejściowego
    input_devices = list_input_devices()
    if not input_devices:
        print("Nie znaleziono urządzeń wejściowych!")
        return

    try:
        dev_index = int(input("\nWybierz urządzenie (numer): "))
        device_id = sd.query_devices().index(input_devices[dev_index])
    except (ValueError, IndexError):
        print("Nieprawidłowy wybór urządzenia!")
        return

    # Parametry nagrania
    duration = float(input("Czas nagrania w sekundach (domyślnie 5): ") or 5)

    # Najlepsze parametry referencyjne
    orig_sr = 96000  # Wysoka częstotliwość próbkowania
    orig_bd = 32  # Wysoka głębokość bitowa (float)

    print(f"\nNagrywam dźwięk referencyjny ({orig_sr} Hz, {orig_bd}-bit float)...")
    high_quality_audio = record_audio(orig_sr, duration, device_id)

    if high_quality_audio is None:
        print("Błąd nagrywania. Program zakończony.")
        return

    # Zapis oryginału
    original_file = "wyniki/original_96kHz_32bit.wav"
    save_audio(original_file, high_quality_audio, orig_sr, orig_bd)

    # Parametry do testowania - szeroki zakres zgodnie z poleceniem
    samplerates = [48000, 44100, 22050, 16000, 11025, 8000]  # Różne częstotliwości
    bit_depths = [24, 16, 12, 8]  # Różne głębokości bitowe

    results = []
    total_combinations = len(samplerates) * len(bit_depths)
    current = 0

    print(f"\nPrzetwarzanie {total_combinations} kombinacji parametrów...")
    print("-" * 50)

    for sr in samplerates:
        for bd in bit_depths:
            current += 1
            print(f"[{current}/{total_combinations}] Przetwarzanie: {sr} Hz, {bd}-bit...")

            # Próbkowanie (resampling)
            audio_resampled = resample_audio(high_quality_audio, orig_sr, sr)

            # Kwantyzacja
            audio_quantized = quantize(audio_resampled, bd)

            # Zapis do pliku
            filename = f"wyniki/audio_{sr}Hz_{bd}bit.wav"
            if save_audio(filename, audio_quantized, sr, bd):
                # Przygotowanie sygnału referencyjnego w tej samej częstotliwości
                reference = resample_audio(high_quality_audio, orig_sr, sr)

                # Obliczenia metryk jakości
                snr = compute_snr(reference, audio_quantized)

                # Wyświetlenie wyników
                print(f"  → SNR: {snr:.2f} dB")

                # Teoretyczne SNR dla porównania
                theoretical_snr = 6.02 * bd + 1.76 if bd <= 24 else float('inf')
                if theoretical_snr != float('inf'):
                    print(f"  → SNR teoretyczne: {theoretical_snr:.2f} dB")

                results.append({
                    'samplerate': sr,
                    'bit_depth': bd,
                    'snr': snr,
                    'filename': filename
                })

    save_results_report(results)

    print("\n" + "=" * 50)
    print("PRZETWARZANIE ZAKOŃCZONE POMYŚLNIE!")
    print("Wszystkie pliki zapisane w katalogu 'wyniki/'")
    print("Sprawozdanie dostępne w pliku 'sprawozdanie.txt'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram przerwany przez użytkownika.")
    except Exception as e:
        print(f"\n\nNiespodziewany błąd: {e}")
        print("Sprawdź połączenia audio i spróbuj ponownie.")