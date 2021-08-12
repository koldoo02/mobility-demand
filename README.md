# Spatio-Temporal Mobility Demand Forecaster (ST-MDF)

Urban mobility is key for the development of any city, particularly in the case of smart cities.
Being able to predict when and where a certain number of people will have the need to start a trip allows for the optimization of any given mobility service, such as taxi, bicycle or for-hire vehicles (FHV).
This repository collects source code and information about a Deep Learning model named ST-MDF for mobility demand forecasting with heterogeneous mobility services.
This forecasting problem can be graphically posed as follows:

![](https://github.com/iipr/mobility-demand/raw/main/graphs/the-problem.png)

The main features of the developed ST-MDF model are that:
- it works with mobility demand mesh-grids, which provides **high spatio-temporal resolution** (500x500m grid with 15min intervals),
- the prediction takes into account `n_x` previous *timesteps*, and forecasts demand for `n_y` **future intervals**,
- it is **robust**, i.e., it can recover from sensor failure and continue producing reliable predictions,
- it is **flexible**, i.e., it can work with a variable number of input and output sensors.

The contents of the repository are described in the following points:
- `graphs`: Relevant graphs about the problem, and several maps of the city chosen as the case of study: Chicago.
- `notebooks`: Jupyter notebooks that present how the data was analyzed and parsed. The obtained datasets are available at: https://doi.org/10.5281/zenodo.5166838
- `results`: Summary that includes metadata and metrics of the trained models and baselines. In the column names, MMWC is the Mean across taxi zones/bike racks of the MAE Weighted of trip Counts, per forecast horizon. Similarly, MRWC is the Mean across taxi zones/bike racks of the RMSE Weighted of trip Counts, per forecast horizon. Also, a learning rate of 0 means that the default value for that Keras optimizer was employed.
- `src`: It includes the following scripts:
  - `deep_playground.py`: Interactive script to train and test deep learning models.
  - `launcher.py`: Script that launches the training and testing of models iteratively.
  - `learner.py`: Main script that works together with `modelUtils.py`, `trainUtils.py` and `plotUtils.py` to manage the training process, while storing the relevant model metadata.
  - `modelUtils.py`: It defines a class that contains the hardcoded models, alongside the relevant model (e.g. `n_x` and `n_y`) and training (e.g. loss function, learning rate, optimizer, etc.) parameters.
  - `plotUtils.py`: It defines a class for plotting tasks.
  - `test_gpu.py`: Script to test whether the GPU/s is/are available.
  - `trainUtils.py`: It defines a data generator class for the training and testing of the models, and a class for keeping the relevant information of the training history.

