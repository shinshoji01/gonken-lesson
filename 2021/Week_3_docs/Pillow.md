# Pillow

## Step1. What is Pillow

Python Imaging Library (abbreviated as PIL) (in newer versions known as Pillow) is a **free and open-source additional library** for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.

**Development appears to be discontinued, with the last commit to the PIL repository coming in 2011.** Consequently, **a successor project called Pillow has forked the PIL repository** and added Python 3.x support.

### Capabilities

Pillow offers **several standard procedures for image manipulation.** These include:

- per-pixel manipulations,
- masking and transparency handling,
- image filtering, such as blurring, contouring, smoothing, or edge finding,
- image enhancing, such as sharpening, adjusting brightness, contrast or color,
- adding text to images and much more.

### File formats

**Some of the file formats supported are PPM, PNG, JPEG, GIF, TIFF, and BMP.**

It is also possible to create new file decoders to expand the library of file formats accessible.

## Step2. Installation

[Pillow official installation documentation](https://pillow.readthedocs.io/en/stable/installation.html#installation)

#### warning

Pillow and PIL cannot co-exist in the same environment. Before installing Pillow, please uninstall PIL.

Pillow >= 1.0 no longer supports “import Image”. Please use “from PIL import Image” instead.

#### versions

Pillow is supported on the following Python versions.

Please check yourself on the bellow page.

https://pillow.readthedocs.io/en/stable/installation.html#notes

```bash
# This is simple and the best installation. 
$ pip install Pillow
```

#### Windows Installation

We provide Pillow binaries for Windows compiled for the matrix of supported Pythons in both 32 and 64-bit versions in the wheel format. These binaries have all of the optional libraries included except for raqm, libimagequant, and libxcb:

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install --upgrade Pillow
```

#### macOS Installation

We provide binaries for macOS for each of the supported Python versions in the wheel format. These include support for all optional libraries except libimagequant and libxcb. Raqm support requires libraqm, fribidi, and harfbuzz to be installed separately:

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install --upgrade Pillow
```

#### Linux Installation

We provide binaries for Linux for each of the supported Python versions in the manylinux wheel format. These include support for all optional libraries except libimagequant. Raqm support requires libraqm, fribidi, and harfbuzz to be installed separately:

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install --upgrade Pillow
```

## Step3. Let's use Pillow

[Pillow official tutorial documentation](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#tutorial)

### Using the Image class

The most important class in the Python Imaging Library is the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image) class, defined in the module with the same name. You can create instances of this class in several ways; either by loading images from files, processing other images, or creating images from scratch.

To load an image from a file, use the [`open()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open) function in the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#module-PIL.Image) module:

```python
from PIL import Image
im = Image.open("hopper.ppm")
```

If successful, this function returns an [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image) object. You can now use instance attributes to examine the file contents:

```python
print(im.format, im.size, im.mode)
# PPM (512, 512) RGB
```

The [`format`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.format) attribute identifies the source of an image. If the image was not read from a file, it is set to None. The size attribute is a 2-tuple containing width and height (in pixels). The [`mode`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.mode) attribute defines the number and names of the bands in the image, and also the pixel type and depth. Common modes are “L” (luminance) for greyscale images, “RGB” for true color images, and “CMYK” for pre-press images.

If the file cannot be opened, an `OSError` exception is raised.

Once you have an instance of the [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image) class, you can use the methods defined by this class to process and manipulate the image. For example, let’s display the image we just loaded:

```python
im.show()
```

### Let's make 2 Python files

```bash
$ mkdir ./Week_03_Pillow # "mkdir" is a makig directory command 
$ cd Week_03_Pillow # "cd" is a changing directory command
$ touch download_cat.py # "touch" is a making file command
$ touch pillow_demo.py
$ atom download_cat.py pillow_demo.py # If you use atom editor
```

```python
# ---download_cat.py--- #
# for downloading cat picture

import os
import pprint
import time
import urllib.error
import urllib.request

def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)
        
url = 'https://1.bp.blogspot.com/-jlHonWZdPp0/Xq5vQuVPQrI/AAAAAAABYtI/S0mjN1WK-wEJBBSS2M6xTEhEmVjM5mUwwCEwYBhgL/s1600/shigoto_zaitaku_cat_man.png'
dst_path = 'cat.png'
download_file(url, dst_path)
```

```python
# ---pillow_demo.py--- #
# for displaying cat picture

from PIL import Image
from PIL import ImageFilter

im = Image.open("cat.png")
im.show()
```

## Step4. Let's play Pillow freely

```python
# Let's add this code to pillow_demo.py

# 1
print(im.format, im.size, im.mode) 
# 2 Select RGB color pixels and maximum value
print(im.getextrema()) 
# 3 Get the color of the specified coordinates. The origin of the coordinates (0, 0) is in the upper left. A tuple of (R, G, B) is returned.
print(im.getpixel((256, 256))) 
# 4 Black-and-white conversion (convert ('L'))
im_gray = im.convert('L')
im_gray.show()
im_gray.save('cat_gray.png', quality=95)
# 5 Rotation by 90 degrees (rotate (90))
im_rotate = im.rotate(90)
im_rotate.show()
im_rotate.save('cat_rotate.png', quality=95)
# 6 Gaussian exclusion is one that has been applied to image smoothing. (ぼかし)
im_gaussianblur = im.filter(ImageFilter.GaussianBlur())
im_gaussianblur.show()
im_gaussianblur.save('cat_gaussianblur.png', quality=95)
```

## Reference

- [Python Imaging Library -wikipedia-](https://en.wikipedia.org/wiki/Python_Imaging_Library)
- [Pillow -official documentation-](https://pillow.readthedocs.io/en/stable/index.html)
- [Pythonの画像処理ライブラリPillow(PIL)の使い方](https://note.nkmk.me/python-pillow-basic/)