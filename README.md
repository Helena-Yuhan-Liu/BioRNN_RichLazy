# BioRNN_RichLazy

This repository ( tested in PyTorch 1.10.2, Torchvision 0.4.2, Neurogym 0.0.2, and Numpy 1.18.1) conducts recurrent neural network (RNN) training across different initial hidden weight effective ranks and logs the associated training laziness measures. The laziness of the learning regime is quantified using hidden weight change norm, representation alignment, and tangent kernel alignment [1-3].

## Usage

The primary script is `main.py`. Helper functions for data storage and retrieval are located in `file_saver_dumper.py`. To execute the main script, use the following command:

``python3 main.py``

Under the default settings, the code should complete in less than 30 minutes.

## References

[1] L Chizat, E Oyallon, and F Bach. On lazy training in differentiable programming. Advances in Neural Information Processing Systems, 32, 2019.

[2] T Flesch, K Juechems, T Dumbalska, A Saxe, and C Summerfield. Orthogonal representations for robust context-dependent task performance in brains and neural networks. Neuron, 110(7):1258–1270, 2022.

[3] T George, G Lajoie, and A Baratin. Lazy vs hasty: linearization in deep networks impacts learning schedule based on example difficulty. arXiv:2209.09658, 2022 

