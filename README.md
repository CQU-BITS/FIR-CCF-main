# FIR-CCF-main
The implementation of knowledge-informed FIR-based Cross-category Filtering (FIR-CCF) framework in Pytorch.
## [knowledge-informed FIR-based Cross-category Filtering (FIR-CCF) framework for Interpretable Machinery Fault Diagnosis under Small samples](https://ieeexplore.ieee.org/document/10443049](https://www.sciencedirect.com/science/article/abs/pii/S0951832024006811)

# Implementation of the paper:
Paper:
```
@article{Interpretable MCN,
  title={Knowledge-informed FIR-based cross-category filtering framework for
interpretable machinery fault diagnosis under small samples},
  author = {Rui Liu and Xiaoxi Ding and Shenglan Liu and Hebin Zheng and Yuanyuan Xu and Yimin Shao},
  journal={Reliability Engineering and System Safety},
  volume = {232},
  pages = {120860},
  year = {2024},
  issn = {0957-4174},
  doi = {10.1016/j.eswa.2023.120860},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417423013623},
}
```
# Requirements
* Python 3.8.8 or newer
* torch-geometric 2.3.1
* pytorch  1.11.0
* numpy  1.23.0

# Guide 
Existing fault diagnosis methods rarely focus on the methodological interpretability and the data scarcity in real industrial scenarios simultaneously. Motivated by this issue, we deeply reexamined the intrinsic characteristics of fault signals and the guiding significance of classical signalprocessing methods for feature enhancement. From the perspective of multiscale modes, this study tailors multiple learnable knowledge-informed finite impulse response (FIR) filtering kernels to extract sensitive modes for explainable feature enhancement. On this foundation, a knowledge-informed FIR-based cross-category filtering (FIR-CCF) framework is further proposed for interpretable small-sample fault diagnosis. With the consideration of the mode complexity, a cross-category filtering strategy is explored to further enhance feature expressions for identifying single state. To be special, this strategy divides a multi-class recognition process into multiple two-class recognition task. A multi-task learning is then presented where multiple binary-class base learners (BCBLearners) that consists of a feature extractor and a two-class classifier is established to seek discriminate mode features for each type of state. Eventually, all feature extractors are fixed and a multi-class classifier is established and to fuse all mode features for high-precision multi-class identification via ensemble learning. As a variant of signal-processing-collaborated deep learning frameworks, the FIR-CCF method fully exploits the strengths of signal-processing methods in interpretability and feature extraction. It can be also foreseen that the signal-processingcollaborated deep learning framework shows enormous potential in interpretable fault diagnosis for knowledge-informed artificial intelligence. 
![MCN](https://github.com/CQU-BITS/MCN-main/blob/main/GA.png)

# Pakages
* `datasets` contians the data load methods for different dataset
* `models` contians the implemented models for diagnosis tasks
* `postprocessing` contians the implemented functions for result visualization

# Datasets
Self-collected datasets
* CQU Gear Dataset (unavailable at present)
### Open source datasets
* [SEU Bearing Dataset](https://github.com/cathysiyu/Mechanical-datasets)
* [MCC5-THU gearbox Dataset](https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets)
* 
# Acknowledgement
* [LaplaceAlexNet](https://github.com/HazeDT/WaveletKernelNet)
* [EWSNet](https://github.com/liguge/EWSNet)
* [LiConvFormer](https://github.com/yanshen0210/LiConvFormer-a-lightweight-fault-diagnosis-framework)
* [MTAGN](https://github.com/shane995/MTAGN)
* 
# Related works
* [R. Liu, X. Ding*, et al., “Knowledge-informed FIR-based cross-category filtering framework for interpretable machinery fault diagnosis under small samples, Reliability Engineering & System Safety, 2024](https://www.sciencedirect.com/science/article/pii/S0951832024006811).
* [R. Liu, X. Ding*, et al., “An Interpretable Multiplication-Convolution Network for Equipment Intelligent Edge Diagnosis, IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024](https://ieeexplore.ieee.org/abstract/document/10443049).
* [R. Liu, X. Ding*, et al., “An interpretable multiplication-convolution residual network for equipment fault diagnosis via time–frequency filtering, Adv. Eng. Inform., 60 (2024) 102421.](https://www.sciencedirect.com/science/article/pii/S1474034624000697)
* [R. Liu, X. Ding*,et al., “Signal processing collaborated with deep learning: An interpretable FIRNet for industrial intelligent diagnosis,” Mech. Syst. Signal Proc., vol. 212, pp. 111314, 2024/04/15/, 2024.](https://www.sciencedirect.com/science/article/pii/S0888327024002127?via%3Dihub#m0005)
* [R. Liu, X. Ding*, et al., “Sinc-Based Multiplication-Convolution Network for Small-Sample Fault Diagnosis and Edge Application,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-12, 2023.](https://ieeexplore.ieee.org/document/10266990)
* [R. Liu, X. Ding*, et al.,Prior-knowledge-guided mode filtering network for interpretable equipment intelligent diagnosis under varying speed conditions, Adv. Eng. Inform., 2024](https://www.sciencedirect.com/science/article/pii/S1474034624001411)
* [Q. Wu, X. Ding*, et al., "An Interpretable Multiplication-Convolution Sparse Network for Equipment Intelligent Diagnosis in Antialiasing and Regularization Constraint," IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-12, 2023.](https://ieeexplore.ieee.org/document/10108914)
* [Q. Wu, X. Ding*, et al., An Intelligent Edge Diagnosis System Based on Multiplication-Convolution Sparse Network, IEEE Sens. J., (2023) 1-1.](https://ieeexplore.ieee.org/document/10227888)

