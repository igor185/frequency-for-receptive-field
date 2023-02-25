# Frequency transformation for image recognition and inpainting
This is repo from final project for course Linear Algebra. You can read final report [here](mid_report.pdf). For checking final score and plots you can use this [colab](https://colab.research.google.com/drive/1Z1VzWt5afsFELqomb8XyGTA8_LaUx7o5?usp=sharing) 

### Installation
```bash
git clone https://github.com/igor185/frequency-for-receptive-field
cd frequency-for-receptive-field
conda env create -f env.yml
conda activate fft
```

### Running

Run la classification
```bash
python la_classification.py
```

Train and evaluate cifar-10
```bash
bash runners/cifar-conv-runner.sh
bash runners/cifar-fourier-runner.sh
bash runners/cifar-wavelet-runner.sh
```


## Results
- Classical classifiers:     
    - SVM on mnist 91% accuracy, SVM on fft from mnist 84%
- Deep learning classifiers:     
    - Network trained on raw pixel performs better then network train on frequencies from pixels    
    - Cifar-10: Network(one resnet block) with fft on deep features performs 3% better then network without fft
- Inpainting results(obtained after running inference from this [repo](https://github.com/advimman/lama) you should follow steps in that repo to obtain same result:   
Input image:
![](lama_results/input_img.png)
Inpainting without fft and dilated conv:
![](lama_results/result_regular.png)
Inpainting with dilated conv:
![](lama_results/result_dilated.png)
Inpainting with fft:
![](lama_results/result_fourier.png)