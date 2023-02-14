# Frequency transformation for image recognition and inpainting

### Steps to do:
- [ ] Test conda env
- [ ] Fourier torch implementation
- [ ] Wavelet torch implementation
- [ ] Finish la classifier
- [ ] Finish code&train conv fft
- [ ] Repeat la classifier and conv training for wavelet 
- [ ] Add results from lama with/without fft

### Installation
```bash
git clone https://github.com/igor185/frequency-for-receptive-field
cd frequency-for-receptive-field
git clone https://github.com/saic-mdal/lama # for comparing inpainting results (can be skipped)
conda env create -f env.yml
conda activate fft
```

### Running

Run la classification
```bash
python la_classification.py
```