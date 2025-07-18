"""
ATuner.py - A simple, efficient tuner application for string instruments like violin.

Features:
- Real-time pitch detection using autocorrelation and parabolic interpolation.
- Analog-style tuning meter visualization with color-coded accuracy zones.
- GUI built with Tkinter for easy interaction and live feedback.
- Device and buffer size selection, A4 reference pitch adjustment.

Main Components:
- PitchDetector: Extracts pitch (Hz) from audio using autocorrelation.
- Tuner: Manages audio input, pitch-to-note conversion, and state.
- Meter: Draws analog-style tuning meter on Tkinter Canvas.
- TunerApp: Main GUI and application logic.

Usage:
- Run this script with Python 3.7+ and required dependencies (numpy, scipy, pyaudio).
- Start/stop tuning, select input device and buffer size, adjust A4 reference pitch.

Dependencies:
- tkinter, numpy, pyaudio, scipy, logging

Notes:
- For best results, use a low-latency audio device and appropriate buffer size.
- The meter highlights violin string notes and provides cent deviation feedback.
"""
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import pyaudio
import math
import queue
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    from scipy.signal import find_peaks, butter, filtfilt
except ImportError:
    messagebox.showerror(
        "Missing Dependencies",
        "SciPy is required. Please install with: pip install scipy"
    )
    exit()

SAMPLE_RATE = 48000
DEFAULT_BUFFER_SIZE = 1024
BUFFER_SIZES = [512, 1024, 2048]
NOTE_STRINGS = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
MAX_CENTS = 50
SMOOTHING_FACTOR = 0.15
POINTER_SMOOTHING = 0.2

VIOLIN_NOTES = {
    'G3': {'frequency': 196.0, 'note_num': 55, 'name': 'G', 'octave': 3},
    'D4': {'frequency': 293.7, 'note_num': 62, 'name': 'D', 'octave': 4},
    'A4': {'frequency': 440.0, 'note_num': 69, 'name': 'A', 'octave': 4},
    'E5': {'frequency': 659.3, 'note_num': 76, 'name': 'E', 'octave': 5}
}

class PitchDetector:
    """
    Detects fundamental frequency using autocorrelation and parabolic interpolation.
    Use detect(audio) to get pitch in Hz from a numpy array.
    - sample_rate: Audio sample rate in Hz
    - buffer_size: Number of samples per analysis frame
    """
    def __init__(self, sample_rate=SAMPLE_RATE, buffer_size=DEFAULT_BUFFER_SIZE):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.middle_a = 440.0
        self.semitone = 69
        self.min_freq = 80
        self.max_freq = 3000
        self.threshold = 0.01

    def _parabolic(self, array, idx):
        """Refine peak index for sub-sample accuracy."""
        if idx < 1 or idx >= len(array) - 1:
            return idx
        y0, y1, y2 = array[idx - 1], array[idx], array[idx + 1]
        denom = 2 * (y0 - 2 * y1 + y2)
        if denom == 0:
            return idx
        return idx + (y0 - y2) / denom

    def detect(self, audio):
        """Returns detected frequency in Hz or None."""
        if len(audio) < self.buffer_size:
            return None
        if np.sqrt(np.mean(audio ** 2)) < self.threshold:
            return None
        nyquist = 0.5 * self.sample_rate
        cutoff = min(3000, self.max_freq * 1.2)
        b, a = butter(4, cutoff / nyquist, btype='low')
        filtered = filtfilt(b, a, audio)
        windowed = filtered * np.hanning(len(filtered))
        autocorr = np.correlate(windowed, windowed, mode='full')[len(windowed)-1:]
        autocorr /= np.max(np.abs(autocorr)) + 1e-10
        min_period = int(self.sample_rate / self.max_freq)
        max_period = int(self.sample_rate / self.min_freq)
        if max_period >= len(autocorr):
            max_period = len(autocorr) - 1
        peaks, _ = find_peaks(autocorr, height=0.1)
        peaks = [p for p in peaks if min_period <= p <= max_period]
        if not peaks:
            return None
        best = max(peaks, key=lambda p: autocorr[p])
        interp = self._parabolic(autocorr, best)
        if interp > 0:
            freq = self.sample_rate / interp
            if self.min_freq <= freq <= self.max_freq:
                return freq
        return None

class Tuner:
    """
    Handles audio input, pitch detection, and note conversion.
    - Use start_recording/stop_recording to control audio stream.
    - Use freq_to_note(freq) to convert frequency to note info.
    - Device selection and buffer size are configurable.
    """
    def __init__(self, sample_rate=SAMPLE_RATE, buffer_size=DEFAULT_BUFFER_SIZE, device_index=None):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.middle_a = 440.0
        self.semitone = 69
        self.note_strings = NOTE_STRINGS
        self.audio = None
        self.stream = None
        self.detector = PitchDetector(sample_rate, buffer_size)
        self.is_recording = False
        self.note_queue = queue.Queue(maxsize=10)
        self.last_note = None
        self.audio_initialized = False
        self.device_index = device_index
        self.signal_level = 0.0

    def list_devices(self):
        """Returns list of available input devices."""
        devices = []
        try:
            audio = pyaudio.PyAudio()
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append((i, info['name']))
            audio.terminate()
        except Exception as e:
            logging.error(f"Device list error: {e}")
        return devices

    def init_audio(self):
        """Initializes audio input stream."""
        try:
            if self.audio_initialized:
                return True
            self.audio = pyaudio.PyAudio()
            if self.device_index is None:
                for i in range(self.audio.get_device_count()):
                    info = self.audio.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        self.device_index = i
                        break
            if self.device_index is None:
                raise Exception("No audio input device found")
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback
            )
            self.audio_initialized = True
            return True
        except Exception as e:
            logging.error(f"Audio error: {e}")
            messagebox.showerror("Audio Error", f"Cannot initialize audio: {str(e)}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Processes audio buffer, updates note_queue."""
        if not self.is_recording:
            return (in_data, pyaudio.paContinue)
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.signal_level = np.sqrt(np.mean(audio_data ** 2))
            freq = self.detector.detect(audio_data)
            if freq:
                note = self.freq_to_note(freq)
                if note:
                    if self.last_note == note['name'] or self.last_note is None:
                        self.last_note = note['name']
                        while not self.note_queue.empty():
                            try:
                                self.note_queue.get_nowait()
                            except queue.Empty:
                                break
                        try:
                            self.note_queue.put_nowait(note)
                        except queue.Full:
                            pass
                    else:
                        self.last_note = note['name']
        except Exception as e:
            logging.warning(f"Audio callback error: {e}")
        return (in_data, pyaudio.paContinue)

    def freq_to_note(self, freq):
        """Converts frequency to note dict (name, octave, cents, etc)."""
        try:
            if freq <= 0:
                return None
            note_num = 12 * math.log2(freq / self.middle_a) + self.semitone
            rounded = round(note_num)
            std_freq = self.middle_a * (2 ** ((rounded - self.semitone) / 12))
            cents = 1200 * math.log2(freq / std_freq)
            cents = max(-MAX_CENTS, min(MAX_CENTS, cents))
            name = self.note_strings[rounded % 12]
            octave = (rounded // 12) - 1
            return {
                'name': name,
                'octave': octave,
                'frequency': freq,
                'cents': cents,
                'note_number': rounded
            }
        except Exception as e:
            logging.warning(f"Note conversion error: {e}")
            return None

    def start_recording(self):
        """Starts audio stream for pitch detection."""
        try:
            if self.stream:
                self.is_recording = True
                if not self.stream.is_active():
                    self.stream.start_stream()
                return True
        except Exception as e:
            logging.warning(f"Start recording error: {e}")
            return False

    def stop_recording(self):
        """Stops audio stream."""
        try:
            self.is_recording = False
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
        except Exception as e:
            logging.warning(f"Stop recording error: {e}")

    def cleanup(self):
        """Releases audio resources."""
        try:
            self.stop_recording()
            if self.stream:
                self.stream.close()
                self.stream = None
            if self.audio:
                self.audio.terminate()
                self.audio = None
            self.audio_initialized = False
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    def set_a4(self, new_pitch):
        """Sets reference pitch for A4."""
        try:
            pitch = float(new_pitch)
            if 400 <= pitch <= 480:
                self.middle_a = pitch
                self.detector.middle_a = pitch
                return True
            else:
                messagebox.showwarning("Invalid Value", "A4 frequency must be between 400.0 and 480.0 Hz")
                return False
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number for A4 frequency")
            return False

class Meter:
    """
    Draws analog-style tuning meter on Tkinter Canvas.
    - update_pointer(cents): update pointer position.
    - create_meter(): draw meter background and scale.
    - Color-coded zones: green (in tune), orange/red (out of tune).
    """
    def __init__(self, canvas, width=440, height=220):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height - 30
        self.radius = min(width, height) * 0.4
        self.pointer_angle = 0
        self.target_angle = 0
        self.create_meter()

    def create_meter(self):
        """Draws meter background and scale."""
        self.canvas.delete("all")
        self.draw_arc(-50, -10, "#FF3030", 8)
        self.draw_arc(-10, -3, "#FF9500", 6)
        self.draw_arc(-3, 3, "#00FF00", 10)
        self.draw_arc(3, 10, "#FF9500", 6)
        self.draw_arc(10, 50, "#FF3030", 8)
        self.draw_scale_marks()
        self.canvas.create_oval(
            self.center_x - 3, self.center_y - 3,
            self.center_x + 3, self.center_y + 3,
            fill="#FFFFFF", outline="#FFFFFF"
        )
        self.pointer_id = None
        self.update_pointer(0)

    def draw_arc(self, start_cents, end_cents, color, width):
        """Draws colored arc segment for cent range."""
        start_angle = 90 - (start_cents * 1.8)
        end_angle = 90 - (end_cents * 1.8)
        extent = end_angle - start_angle
        self.canvas.create_arc(
            self.center_x - self.radius, self.center_y - self.radius,
            self.center_x + self.radius, self.center_y + self.radius,
            start=start_angle, extent=extent,
            style="arc", outline=color, width=width
        )

    def draw_scale_marks(self):
        """Draws major and minor cent scale marks."""
        for cents in range(-50, 51, 10):
            angle = math.radians(90 + cents * 1.8)
            self.draw_tick(angle, 15, 2, str(cents) if cents != 0 else "0")
        for cents in range(-45, 46, 10):
            if cents % 10 != 0:
                angle = math.radians(90 + cents * 1.8)
                self.draw_tick(angle, 10, 1)

    def draw_tick(self, angle, length, width, label=None):
        """Draws single tick mark and optional label."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        start_x = self.center_x - (self.radius - length) * cos_a
        start_y = self.center_y - (self.radius - length) * sin_a
        end_x = self.center_x - self.radius * cos_a
        end_y = self.center_y - self.radius * sin_a
        self.canvas.create_line(
            start_x, start_y, end_x, end_y,
            fill="#FFFFFF", width=width
        )
        if label:
            label_x = self.center_x - (self.radius + 20) * cos_a
            label_y = self.center_y - (self.radius + 20) * sin_a
            self.canvas.create_text(
                label_x, label_y, text=label,
                font=("Arial", 8, "bold"), fill="#FFFFFF"
            )

    def update_pointer(self, cents):
        """Updates pointer position for cent deviation."""
        if self.pointer_id:
            self.canvas.delete(self.pointer_id)
        self.target_angle = math.radians(90 + np.clip(cents, -MAX_CENTS, MAX_CENTS) * 1.8)
        self.pointer_angle += (self.target_angle - self.pointer_angle) * POINTER_SMOOTHING
        pointer_length = self.radius * 0.8
        end_x = self.center_x - pointer_length * math.cos(self.pointer_angle)
        end_y = self.center_y - pointer_length * math.sin(self.pointer_angle)
        self.pointer_id = self.canvas.create_line(
            self.center_x, self.center_y, end_x, end_y,
            fill="#FF0000", width=4, capstyle="round"
        )

class TunerApp:
    """
    Main GUI application. Handles user interaction and display updates.
    - create_widgets(): sets up all UI widgets.
    - update_display(): updates all UI elements with latest tuning and signal data.
    - on_device_change/on_buffer_change: handle device/buffer selection.
    - open_settings(): dialog for A4 pitch.
    - toggle_tuning(): start/stop tuning.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Tuner")
        self.root.geometry("420x750")
        self.root.configure(bg='#1a1a1a')
        self.root.resizable(True, True)
        self.colors = {
            'bg': '#1a1a1a',
            'fg': '#ffffff',
            'secondary': '#888888',
            'perfect': '#00ff00',
            'good': '#32d74b',
            'warning': '#ff9f0a',
            'error': '#ff453a',
            'button_active': '#0a84ff',
            'button_inactive': '#333333',
        }
        self.tuner = Tuner()
        self.current_note = None
        self.is_running = False
        self.smooth_cents = 0
        self.last_update_time = time.time()
        self.selected_buffer_size = tk.IntVar(value=DEFAULT_BUFFER_SIZE)
        self.selected_device = tk.IntVar(value=0)
        self.smooth_signal_level = 0.0
        self.create_widgets()
        self.start_update_loop()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Sets up all UI widgets."""
        settings_frame = tk.Frame(self.root, bg=self.colors['bg'])
        settings_frame.pack(pady=12)
        tk.Label(settings_frame, text="Input Device:", bg=self.colors['bg'], fg=self.colors['secondary']).pack(side=tk.LEFT)
        devices = self.tuner.list_devices()
        device_names = [name for idx, name in devices]
        self.device_menu = ttk.Combobox(settings_frame, values=device_names, state="readonly", width=25)
        self.device_menu.current(0)
        self.device_menu.pack(side=tk.LEFT, padx=5)
        self.device_menu.bind("<<ComboboxSelected>>", self.on_device_change)
        tk.Label(settings_frame, text="Buffer Size:", bg=self.colors['bg'], fg=self.colors['secondary']).pack(side=tk.LEFT, padx=(10,0))
        self.buffer_menu = ttk.Combobox(settings_frame, values=BUFFER_SIZES, state="readonly", width=6)
        self.buffer_menu.current(BUFFER_SIZES.index(DEFAULT_BUFFER_SIZE))
        self.buffer_menu.pack(side=tk.LEFT)
        self.buffer_menu.bind("<<ComboboxSelected>>", self.on_buffer_change)
        self.note_frame = tk.Frame(self.root, bg=self.colors['bg'])
        self.note_frame.pack(pady=10)
        self.note_label = tk.Label(
            self.note_frame, text="--",
            font=("Arial", 60, "bold"),
            bg=self.colors['bg'], fg=self.colors['fg']
        )
        self.note_label.pack(side=tk.LEFT, anchor='s')
        self.octave_label = tk.Label(
            self.note_frame, text="",
            font=("Arial", 20, "bold"),
            bg=self.colors['bg'], fg=self.colors['secondary']
        )
        self.octave_label.pack(side=tk.LEFT, anchor='n', padx=5, pady=15)
        self.meter_canvas = tk.Canvas(
            self.root, width=440, height=220,
            bg=self.colors['bg'], highlightthickness=0
        )
        self.meter_canvas.pack(pady=10)
        self.meter = Meter(self.meter_canvas, width=440, height=220)
        info_frame = tk.Frame(self.root, bg=self.colors['bg'])
        info_frame.pack(pady=15, fill=tk.X, padx=30)
        self.frequency_label = tk.Label(
            info_frame, text="0.0 Hz",
            font=("Arial", 14),
            bg=self.colors['bg'], fg=self.colors['secondary']
        )
        self.frequency_label.pack(side=tk.LEFT)
        self.cents_label = tk.Label(
            info_frame, text="0¢",
            font=("Arial", 14, "bold"),
            bg=self.colors['bg'], fg=self.colors['secondary']
        )
        self.cents_label.pack(side=tk.RIGHT)
        self.signal_bar = tk.Canvas(self.root, width=360, height=18, bg=self.colors['bg'], highlightthickness=0)
        self.signal_bar.pack(pady=8)
        self.status_label = tk.Label(
            self.root, text="Ready",
            font=("Arial", 16, "bold"),
            bg=self.colors['bg'], fg=self.colors['secondary']
        )
        self.status_label.pack(pady=10)
        button_frame = tk.Frame(self.root, bg=self.colors['bg'])
        button_frame.pack(pady=20, fill=tk.X, padx=30)
        self.start_button = tk.Button(
            button_frame, text="Start Tuning",
            font=("Arial", 16, "bold"),
            bg=self.colors['button_active'], fg=self.colors['fg'],
            activebackground="#0060c0", activeforeground=self.colors['fg'],
            relief=tk.FLAT, command=self.toggle_tuning
        )
        self.start_button.pack(fill=tk.X, ipady=12, pady=(0, 8))
        settings_button = tk.Button(
            button_frame, text="Settings",
            font=("Arial", 12),
            bg=self.colors['button_inactive'], fg=self.colors['fg'],
            activebackground="#555555", activeforeground=self.colors['fg'],
            relief=tk.FLAT, command=self.open_settings
        )
        settings_button.pack(fill=tk.X, ipady=8)

    def on_device_change(self, event):
        """Handles input device selection change."""
        idx = self.device_menu.current()
        self.tuner.device_index = idx
        self.tuner.cleanup()
        self.tuner = Tuner(buffer_size=self.tuner.buffer_size, device_index=idx)
        logging.info(f"Changed input device to {idx}")

    def on_buffer_change(self, event):
        """Handles buffer size selection change."""
        size = int(self.buffer_menu.get())
        self.tuner.buffer_size = size
        self.tuner.detector.buffer_size = size
        self.tuner.cleanup()
        self.tuner = Tuner(buffer_size=size, device_index=self.tuner.device_index)
        logging.info(f"Changed buffer size to {size}")

    def open_settings(self):
        """Opens dialog to set A4 pitch."""
        new_pitch = simpledialog.askstring(
            "A4 Pitch Setting",
            f"Current A4 pitch: {self.tuner.middle_a:.1f} Hz\n"
            f"Enter new value (400-480):",
            parent=self.root
        )
        if new_pitch:
            if self.tuner.set_a4(new_pitch):
                messagebox.showinfo("Success", f"A4 pitch set to {self.tuner.middle_a} Hz")

    def toggle_tuning(self):
        """Starts or stops tuning."""
        if not self.is_running:
            if self.tuner.init_audio():
                if self.tuner.start_recording():
                    self.is_running = True
                    self.start_button.config(text="Stop Tuning", bg=self.colors['error'])
                    self.status_label.config(text="Listening...", fg=self.colors['button_active'])
                else:
                    messagebox.showerror("Error", "Cannot start recording")
            else:
                messagebox.showerror("Error", "Cannot initialize audio")
        else:
            self.stop_tuning()

    def stop_tuning(self):
        """Stops tuning and resets display."""
        self.is_running = False
        self.tuner.stop_recording()
        self.start_button.config(text="Start Tuning", bg=self.colors['button_active'])
        self.reset_display()

    def reset_display(self):
        """Resets all display elements."""
        self.status_label.config(text="Ready", fg=self.colors['secondary'])
        self.note_label.config(text="--", fg=self.colors['fg'])
        self.octave_label.config(text="")
        self.frequency_label.config(text="0.0 Hz")
        self.cents_label.config(text="0¢")
        self.current_note = None
        self.smooth_cents = 0
        self.signal_bar.delete("all")

    def start_update_loop(self):
        """Starts periodic UI update loop."""
        self.update_display()
        self.root.after(17, self.start_update_loop)

    def update_display(self):
        """Updates all UI elements with latest tuning and signal data."""
        try:
            current_time = time.time()
            if self.is_running:
                latest_note = None
                while not self.tuner.note_queue.empty():
                    try:
                        latest_note = self.tuner.note_queue.get_nowait()
                    except queue.Empty:
                        break
                if latest_note:
                    self.current_note = latest_note
                    self.last_update_time = current_time
                    self.note_label.config(text=self.current_note['name'])
                    self.octave_label.config(text=str(self.current_note['octave']))
                    self.frequency_label.config(text=f"{self.current_note['frequency']:.1f} Hz")
                    target_cents = np.clip(self.current_note['cents'], -MAX_CENTS, MAX_CENTS)
                    self.smooth_cents += (target_cents - self.smooth_cents) * SMOOTHING_FACTOR
                    self.cents_label.config(text=f"{self.smooth_cents:+.1f}¢")
                    abs_cents = abs(self.smooth_cents)
                    if abs_cents <= 2:
                        status, color = "Perfect!", self.colors['perfect']
                    elif abs_cents <= 5:
                        status, color = "Excellent", self.colors['good']
                    elif abs_cents <= 10:
                        status, color = "Good", self.colors['warning']
                    else:
                        status = "Sharp" if self.smooth_cents > 0 else "Flat"
                        color = self.colors['error']
                    is_violin_string = any(
                        note_info['name'] == self.current_note['name'] and 
                        note_info['octave'] == self.current_note['octave']
                        for note_info in VIOLIN_NOTES.values()
                    )
                    if is_violin_string:
                        status = f"♪ {status}"
                    self.status_label.config(text=status, fg=color)
                    self.note_label.config(fg=color)
                    self.cents_label.config(fg=color)
                elif current_time - self.last_update_time > 1.0:
                    self.status_label.config(text="No Signal", fg=self.colors['secondary'])
                    self.note_label.config(fg=self.colors['secondary'])
                    self.cents_label.config(fg=self.colors['secondary'])
            display_cents = self.smooth_cents if self.is_running else 0
            self.meter.update_pointer(display_cents)
            self.signal_bar.delete("all")
            target_level = min(1.0, self.tuner.signal_level * 10)
            self.smooth_signal_level += (target_level - self.smooth_signal_level) * 0.15
            fill = self.colors['good'] if self.smooth_signal_level > 0.2 else self.colors['warning']
            bar_length = int(360 * self.smooth_signal_level)
            if bar_length > 0:
                self.signal_bar.create_rectangle(0, 0, bar_length, 18, fill=fill, outline='')
        except Exception as e:
            logging.warning(f"Update display error: {e}")

    def on_closing(self):
        """Handles application close and cleanup."""
        try:
            self.is_running = False
            self.tuner.cleanup()
            self.root.destroy()
        except Exception as e:
            logging.warning(f"On closing error: {e}")

if __name__ == "__main__":
    # Entry point: launches the tuner GUI
    try:
        root = tk.Tk()
        app = TunerApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application Error: {e}")
        messagebox.showerror("Application Error", f"Cannot start application: {e}")
