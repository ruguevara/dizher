# Dizher

Dizher (and AvtoDizher) is a 8-bit Graphics Converter for lovely retrocomputers with color-restricive graphics modes.

Turn on, tune in, drop out.

## Supported 8-bit platforms and modes

* ZX Spectrum
  * Standard mode — 256x192 with 15 colors, two colors in one character 8x8 block. No flashing bit support.

## Installation

Requires Python 3.10+. From the repository root:

    python3.13 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .

Run the GUI:

    dizher [image.png]

## TODO

* [ ] Correct Error-Diffusion dithering respecting character blocks color restrictions
* [ ] Adjustable metrics weights
* [ ] Selectable dithering methods
* [ ] Brightness, Contrast, Saturation and Vibe controls
* [ ] Save as SCR and PNG
* [ ] Crop adjustments
* [ ] Python installation package
* [ ] Commodore 64 HiRes 320x200 mode with 2 colors per character block
* [ ] Custom ZX Spectrum palettes (ZX Spectrum Next or another hardware enhanceds)
* [ ] ZX Spectrum MultiColor, GigaScreen and MultiGigaScreen software mode
* [ ] Commodore 64 LowRes 160x200 mode with 4 colors per character block
* [ ] Standalone binary package with PyInstaller for macOS, Windows, Linux
* [ ] Overpaint color brushes
* [ ] Ability to save and load conversion projects with an image, settings and overpaint layers
* [ ] Kivy GUI
* [ ] Android application
* [ ] iOS application
