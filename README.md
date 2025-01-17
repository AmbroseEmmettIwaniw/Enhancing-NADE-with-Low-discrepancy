# Enhancing-NADE-with-Low-discrepancy

First, you would need to create a folder called dataset.

For modelling the Binarized MNIST dataset, we use the ConvNADE-Bernoulli model. We used the built-in Pytorch dataset MNIST, which we stochastically binarized. The ConvNADE-Bernoulli model is in the file convnade_bernolli.py and the model is used to model the Binarized MNIST using the run_convnade_bernoulli.py file.

For modelling the CIFAR-10 dataset, we use the ConvNADE-Beta-Color model. We used the built-in Pytorch dataset CIFAR-10. The ConvNADE-Beta-Color model is in the file convnade_beta_color.py and the model is used to model the CIFAR-10 using the run_convnade_beta_color.py file.

For modelling the FER2013 dataset, we use the ConvNADE-Beta-Color model. We used the dataset FER2013, which was downloaded into the dataset folder from https://www.kaggle.com/datasets/msambare/fer2013. The ConvNADE-Beta-Color model is in the file convnade_beta_color.py and the model is used to model the FER2013 using the run_convnade_beta_color.py file.

For modelling the LHQ dataset, we use the ConvNADE-Beta-Color model. We used the dataset LHQ, which was downloaded into the dataset folder from https://github.com/universome/alis/blob/master/lhq.md choose the LHQ256 and download from there. The ConvNADE-Beta-Color model is in the file convnade_beta_color.py and the model is used to model the LHQ using the run_convnade_beta_color.py file.
