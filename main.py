import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import os


def record_audio(filename, samplerate, duration, bit_depth):
    print(f"Nagrywanie {duration} sekund dźwięku z częstotliwością {samplerate} Hz i {bit_depth}-bitową kwantyzacją...")
    try:
        # sd.rec() użyje sd.default.device, które zostało ustawione w select_input_device()
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
    except sd.PortAudioError as e:
        current_input_device_name = "nieznane"
        try:
            # sd.default.device jest krotką (input_idx, output_idx)
            # input_idx to sd.default.device[0]
            if sd.default.device is not None and sd.default.device[0] is not None:
                current_input_device_name = sd.query_devices(sd.default.device[0])['name']
        except Exception:
            # Ignoruj błędy przy pobieraniu nazwy urządzenia, jeśli sd.default.device nie jest ustawione zgodnie z oczekiwaniami
            pass

        print(f"BŁĄD: Nie można otworzyć strumienia wejściowego audio na urządzeniu '{current_input_device_name}'.")
        if hasattr(e, 'pa_errorcode') and e.pa_errorcode == -9997:  # PaErrorCode -9997 to Invalid sample rate
            print(f"Przyczyna: Nieprawidłowa częstotliwość próbkowania ({samplerate} Hz) dla tego urządzenia.")
            print(
                f"Sprawdź, czy urządzenie '{current_input_device_name}' obsługuje {samplerate} Hz, lub wybierz inne urządzenie.")
        else:
            print(f"Szczegóły błędu PortAudio: {e}")
        print("Nagrywanie nie powiodło się.")
        return False  # Sygnalizuj błąd
    except Exception as e:  # Inne możliwe błędy
        print(f"Wystąpił nieoczekiwany błąd podczas nagrywania: {e}")
        return False

    max_val = 2 ** (bit_depth - 1) - 1
    quantized_audio = np.int16(audio_data[:, 0] * max_val)  # Zakładamy mono, audio_data[:, 0]
    try:
        write(filename, samplerate, quantized_audio)
        print(f"Zapisano plik: {filename}")
        return True  # Sygnalizuj sukces
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku {filename}: {e}")
        return False


def play_audio(filename):
    try:
        samplerate, data = read(filename)
        if data.size == 0:
            print(f"Plik {filename} jest pusty. Pomijanie odtwarzania.")
            return

        # Normalizacja tylko jeśli max_abs_val nie jest zerem
        max_abs_val = np.max(np.abs(data))
        if max_abs_val > 0:
            data_float = data.astype(np.float32) / max_abs_val
        else:
            data_float = data.astype(np.float32)  # Dane są zerami

        print(f"Odtwarzanie pliku: {filename}")
        sd.play(data_float, samplerate)
        sd.wait()
    except FileNotFoundError:
        print(f"BŁĄD: Plik {filename} nie został znaleziony. Nie można odtworzyć.")
    except Exception as e:
        print(f"BŁĄD podczas odtwarzania pliku {filename}: {e}")


def compute_snr(reference, test):
    # Upewnij się, że tablice mają ten sam kształt i nie są puste
    if reference.shape != test.shape or reference.size == 0:
        print("Błąd: Tablice referencyjna i testowa mają różne kształty lub są puste. Nie można obliczyć SNR.")
        return float('-inf')  # Lub inna wartość błędu

    noise = reference - test
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 0:  # Jeśli szum jest zerowy lub ujemny (co nie powinno się zdarzyć dla mocy)
        # Może oznaczać idealne dopasowanie lub problem z danymi
        if signal_power > 0:  # Jeśli jest sygnał, a nie ma szumu, SNR jest nieskończony
            return float('inf')
        else:  # Jeśli nie ma sygnału i nie ma szumu, SNR jest nieokreślony, można zwrócić 0 lub NaN
            return 0.0  # Lub float('nan')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def select_input_device():
    devices = sd.query_devices()
    input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]

    if not input_devices:
        print("BŁĄD: Nie znaleziono żadnych urządzeń wejściowych. Upewnij się, że mikrofon jest podłączony.")
        return False  # Sygnalizuj błąd

    print("Dostępne urządzenia wejściowe:")
    for idx, dev in enumerate(input_devices):
        print(f"{idx}: {dev['name']} (domyślna częstotliwość próbkowania: {dev.get('default_samplerate', 'N/A')} Hz)")

    while True:
        try:
            selection_str = input(f"Wybierz numer mikrofonu (0-{len(input_devices) - 1}): ")
            if not selection_str:  # Pusty input
                print("Nie dokonano wyboru. Spróbuj ponownie.")
                continue
            selection = int(selection_str)
            if 0 <= selection < len(input_devices):
                selected_device_info = input_devices[selection]
                # Znajdź indeks wybranego urządzenia w pełnej liście urządzeń `devices`
                # Metoda .index() na liście słowników może być zawodna, jeśli słowniki nie są identyczne.
                # Bezpieczniej jest iterować i porównywać unikalny identyfikator, jeśli dostępny, lub kombinację pól.
                # Dla uproszczenia, zakładamy, że obiekt selected_device_info jest tym samym obiektem, który znajduje się w `devices`.
                try:
                    original_device_index = devices.index(selected_device_info)
                except ValueError:
                    # Jeśli powyższe zawiedzie, spróbuj znaleźć po nazwie i hostapi (bardziej niezawodne)
                    original_device_index = -1
                    for i, dev_item in enumerate(devices):
                        if dev_item['name'] == selected_device_info['name'] and \
                                dev_item['hostapi'] == selected_device_info['hostapi'] and \
                                dev_item['max_input_channels'] == selected_device_info['max_input_channels']:
                            original_device_index = i
                            break
                    if original_device_index == -1:
                        print("Błąd wewnętrzny: Nie można zmapować wybranego urządzenia na listę systemową.")
                        return False

                sd.default.device = (original_device_index, None)  # Ustawia tylko urządzenie wejściowe
                print(f"Używany mikrofon: {selected_device_info['name']}")
                return True  # Sygnalizuj sukces
            else:
                print(f"Nieprawidłowy wybór. Wprowadź liczbę z zakresu 0-{len(input_devices) - 1}.")
        except ValueError:
            print("Nieprawidłowe wejście. Proszę wprowadzić liczbę.")
        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd podczas wyboru urządzenia: {e}")
            return False


def main():
    if not select_input_device():
        print("Nie udało się wybrać urządzenia wejściowego lub żadne nie jest dostępne. Przerywanie programu.")
        return

    duration = 5  # sekundy
    samplerates = [8000, 16000, 44100]
    bit_depths = [8, 16]

    os.makedirs("wyniki", exist_ok=True)

    # Nagrywanie wersji referencyjnej (najwyższe parametry)
    ref_file = "wyniki/ref.wav"
    print("\n--- Nagrywanie wersji referencyjnej ---")
    if not record_audio(ref_file, 44100, duration, 16):
        print(f"Nie udało się nagrać pliku referencyjnego ({ref_file}).")
        print("Dalsza analiza SNR nie jest możliwa bez pliku referencyjnego. Zamykanie programu.")
        return

    try:
        ref_sr, ref_data = read(ref_file)
    except FileNotFoundError:
        print(f"Krytyczny błąd: Plik referencyjny {ref_file} nie został znaleziony. Przerywanie.")
        return
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku referencyjnego {ref_file}: {e}. Przerywanie.")
        return

    if ref_data.size == 0:
        print(f"Plik referencyjny {ref_file} jest pusty. Przerywanie programu.")
        return
    ref_data = ref_data.astype(np.float32)

    for sr in samplerates:
        for bd in bit_depths:
            filename = f"wyniki/audio_{sr}Hz_{bd}bit.wav"
            print(f"\n--- Nagrywanie i analiza dla {sr}Hz, {bd}bit ---")
            if not record_audio(filename, sr, duration, bd):
                print(f"Pominięto obliczanie SNR dla {sr}Hz, {bd}bit z powodu błędu nagrywania.")
                continue

            try:
                sr_test, data_test_raw = read(filename)
            except FileNotFoundError:
                print(f"Plik testowy {filename} nie został utworzony lub znaleziony. Pomijanie.")
                continue
            except Exception as e:
                print(f"Błąd podczas wczytywania pliku testowego {filename}: {e}. Pomijanie.")
                continue

            if data_test_raw.size == 0:
                print(f"Plik testowy {filename} jest pusty. Pomijanie SNR.")
                snr = float('-inf')
            else:
                # Normalizacja danych testowych
                max_abs_val_test = np.max(np.abs(data_test_raw))
                if max_abs_val_test > 0:
                    data_test = data_test_raw.astype(np.float32) / max_abs_val_test
                else:  # Jeśli dane są zerowe
                    data_test = data_test_raw.astype(np.float32)

                # Interpolacja danych referencyjnych do długości danych testowych
                # Upewnij się, że ref_data i data_test mają sensowne długości do interpolacji
                if len(ref_data) == 0 or len(data_test) == 0:
                    print("Dane referencyjne lub testowe są puste po normalizacji. Pomijanie SNR.")
                    snr = float('-inf')
                else:
                    # Tworzenie osi czasu dla interpolacji
                    x_ref = np.linspace(0, 1, num=len(ref_data), endpoint=False)
                    x_test = np.linspace(0, 1, num=len(data_test), endpoint=False)

                    # Interpolacja sygnału referencyjnego do osi czasu sygnału testowego
                    ref_interp = np.interp(x_test, x_ref, ref_data)

                    # Upewnij się, że znormalizowany sygnał referencyjny ma tę samą skalę co testowy (-1, 1)
                    # Jeśli ref_data był już znormalizowany, to ok. Jeśli nie, znormalizuj ref_interp.
                    # Zakładając, że ref_data (po odczycie z pliku WAV) nie jest znormalizowany do [-1, 1] tak jak data_test
                    max_abs_ref_interp = np.max(np.abs(ref_interp))
                    if max_abs_ref_interp > 0:
                        ref_interp_normalized = ref_interp / max_abs_ref_interp
                    else:
                        ref_interp_normalized = ref_interp  # same zera

                    snr = compute_snr(ref_interp_normalized, data_test)  # Użyj znormalizowanego ref_interp

            print(f"SNR dla {sr}Hz, {bd}bit: {snr:.2f} dB")
            play_audio(filename)


if __name__ == "__main__":
    main()