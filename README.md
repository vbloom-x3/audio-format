# BLOS Audio Codec

![GitHub License](https://img.shields.io/github/license/vbloom-x3/audio-format)  ![Build](https://img.shields.io/badge/build-passing-brightgreen)  ![Platform](https://img.shields.io/badge/platform-linux%20%7C%20unix-blue)  ![Language](https://img.shields.io/badge/language-C-lightgrey)

**BLOS** is a minimal, experimental, **single-file audio format and codec written in C**.
It achieves compression ratios of **5–6%** on 16-bit WAV files, while staying simple and hackable.

---

## Why BLOS?

Most modern audio codecs are **big, complex, and dependency-heavy**. BLOS takes the opposite route:

* A **single C file** you can read in one sitting.
* A format that’s **easy to inspect, tinker with, and extend**.
* Compression that’s not the tightest on Earth, but **transparent and surprisingly efficient** for its size.

It’s designed as a playground for learning about **LPC, residual coding, and audio compression basics**, while still being usable for actual audio storage.

---

## Features

* **Single C file** – no external dependencies beyond `libsndfile` and `math`.
***LPC compression** with Levinson-Durbin algorithm.
* **Rice coding** for residuals.
* **Mid/Side stereo coding** for efficient stereo compression.
* **Transient bypass** for large residuals (stores raw samples when necessary).
* **Frame-based, bit-packed, aligned** stream format.
* Compression ratios around **5–6%** on typical audio material.

---

## Building

Make sure you have **libsndfile** installed.

On Linux:

```bash
sudo apt install libsndfile1-dev   # Debian/Ubuntu
# or
sudo pacman -S libsndfile          # Arch
```

Then clone and build:

```bash
git clone https://github.com/vbloom-x3/audio-format
cd audio-format
make
```

This will produce the executable:

```
./codec
```

---

##Usage

### Encode a WAV file

To compress a raw WAV file into `.blos` format:

```bash
./codec input.wav output.blos
```

**Example:**

Original waveform (`input.wav`):

![Original WAV](assets/encoded.png)

---

### Decode a BLOS file

To decompress a `.blos` file back into a playable WAV:

```bash
./codec -d input.blos output.wav
```

**Example:**

Decoded waveform (`output.wav`):

![Decoded WAV](assets/decoded.png)

---

## File Format Overview

* **Magic**: `"BLOS"`
* **Header**: sample rate, frame count, channel count, frame size.
* **Frames**:

  * Seeds (initial samples for prediction).
  * Float64 LPC coefficients.
  * Residuals encoded with **Rice coding**.
  * Escapes for transient bypass.

---

## Example Output

During encoding you’ll see progress and per-frame stats:

```
[Encode] frames=1234 frame_size=512
Frame 1/1234: order=12 kA=4 kB=5 meanA=8.2 meanB=7.5 escA=0 escB=2
Encoded frame 1/1234
...
[Encode] done: output.blos
```

---

## License

This project is licensed under the **GNU General Public License v3.0**.
You may redistribute and/or modify it under the terms of the GPLv3.

For details, see the [LICENSE](LICENSE) file.

---

## Notes

* This is **experimental**—expect quirks and possible artifacts.
* Future improvements might include entropy coding, adaptive thresholds, and better stereo handling.

---
