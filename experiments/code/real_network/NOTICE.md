# Network architecture attribution

The grayscale DRUNet and original FFDNet architectures in `denoisers.py`
are adapted from Kai Zhang's [KAIR](https://github.com/cszn/KAIR),
specifically `models/network_unet.py`, `models/network_ffdnet.py`, and
`models/basicblock.py`. The DeepInv FFDNet layout follows its documented
[converted weights](https://huggingface.co/deepinv/FFDNet) and
[implementation](https://github.com/deepinv/deepinv/blob/main/deepinv/models/ffdnet.py).
No pretrained weights or data are included in this source directory.

MIT License

Copyright (c) 2019 Kai Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
