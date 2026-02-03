# Validation Strategies for Deep Learning-Based Groundwater Level Time Series Prediction Using Exogenous Meteorological Input Features

This repository contains the necessary code to train and validate the 1D-CNN or LSTM model from the submitted paper:

Doll, F., Liesch, T., Wetzel, M., Kunz, S., Borda, S.: Validation Strategies for Deep Learning-Based Groundwater Level Time Series Prediction Using Exogenous Meteorological Input Features

Contact: Fabienne.Doll@kit.edu

ORCIDs of authors:

- F. Doll: 0009-0003-5455-7162
- T. Liesch: 0000-0001-8648-5333
- M. Wetzel: 0000-0002-2289-2156
- S. Kunz: 0000-0001-7074-1865
- S. Broda: 0000-0001-6858-6368

The scripts stored here were used in an Anaconda environment on Tensorfow running with GPU-Support. To reproduce the corresponding environment, the instructions in Create_Environment.txt should be followed.

Necessary codes for training, validating and testing the models can be found under validate_and_test_model. The error metrics can be calculated with the codes under Evaluate_results. Paths to the result folders or data must be adjusted in the code depending on the folder structure.


The results presented in the paper are stored in the zip archive Results_1DCNN_LSTM.zip.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).  
© 2025 KIT Hydrogeology. Commercial use is not permitted.

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


---

For questions or contributions, please open an issue on this GitHub repository.
