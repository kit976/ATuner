# Tuner: A Real-Time Instrument Tuning App

A real-time, cross-platform pitch detection and instrument tuning application built with Python and Tkinter. It features a sleek graphical user interface and audio signal processing pipeline for accurate tuning of instruments like violin, guitar, and voice.

---

## Features

- 🎧 Real-time pitch detection using autocorrelation
- 🎨 Interactive and responsive analog meter
- 🔄 Supports multiple buffer sizes and input devices
- 🎻 Intelligent status feedback (Perfect, Sharp, Flat)
- 🎼 Adjustable A4 tuning reference (400–480 Hz)

---

## Installation

### Requirements

- Python 3.10
- `numpy`
- `pyaudio`
- `scipy`
- `tkinter` (included by default with most Python installations)

### Install Dependencies

```bash
pip install numpy pyaudio scipy
```

> If you're using a virtual environment, activate it first.

---

## Usage

```bash
python ATuner.py
```

- Select your input device (microphone) and buffer size
- Press **"Start Tuning"** to begin
- Optional: Click **"Settings"** to customize the reference A4 pitch

---

## Supported Instruments

This tuner is particularly optimized for violin, detecting:

- G3 (196.0 Hz)
- D4 (293.7 Hz)
- A4 (440.0 Hz)
- E5 (659.3 Hz)

But it also works for any instrument in the ~80–3000 Hz range.

---

## Project Structure

```
.
├── ATuner.py           # Main application file (GUI + logic)
├── README.md      # Documentation
└── LICENSE        # BSD-3 License
```

---

## License

This project is licensed under the BSD-3-Clause License. See [`LICENSE`](LICENSE) for details.

---

## Acknowledgments

- Inspired by classical analog tuning meters
- Thanks to the open-source community for `PyAudio`, `NumPy`, `SciPy`, and `Tkinter`
