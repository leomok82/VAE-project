# Transforming Wildfire Management: Advanced Wildfire Prediction with LSTMs and Dynamic Image Generation through VAE and Data Assimilation 🔥

## Project overview

In this project, we focused on enhancing wildfire prediction and visualization using advanced machine learning techniques. By integrating Recurrent Neural Networks (RNN), Generative AI models, and Data Assimilation methods, we aimed to create a comprehensive framework for accurate wildfire forecasting and dynamic visualization. The project was structured around three main objectives, each contributing to the overall goal of improving wildfire management through innovative AI-driven approaches.

### Objective 1: Building a Surrogate Model with LSTMs to predict wildfire behaviour 🔍

Steps:

 - Train the RNN model using wildfire predictive model data.
 - Use the trained RNN model with background data (Ferguson_fire_background) to make forecasts.
 - Compare the forecasted results with satellite data (Ferguson_fire_obs) and compute the Mean Squared Error (MSE) between the forecast and the satellite observations.


### Objective 2: Building a Surrogate Model with Generative AI to capture wildfire dynamics 🌲🔥


Steps:
 - Train a generative model using training data (Ferguson_fire_train) and test it with testing data (Ferguson_fire_test).
 - Use the trained wildfire generative model to make forecasts.
- Compare the forecasted results with satellite data (Ferguson_fire_obs) and compute the MSE.



### Objective 3: Perform Data Assimilation to improve the accuracy of wildfire predictions using the outputs from the RNN and Generative models 📊🛰️
Steps:
 - Compute the error covariance matrices for background data (matrix B) and satellite data (matrix R). The observation error covariance matrix R must be computed using satellite data.
 - Conduct data assimilation in a reduced space, utilizing the satellite and the generated images.

-------------------------

## Repository structure 📁

```
├── CAE.ipynb
├── Chosing_Image.ipynb
├── README.md
├── WildfireThomas
│ ├── WildfireDA
│ │ ├── init.py
│ │ ├── models
│ │ │ ├── CAEmodel.py
│ │ │ └── init.py
│ │ ├── task3functions
│ │ │ ├── init.py
│ │ │ └── assimilate.py
│ │ └── tests
│ │ ├── init.py
│ │ └── test_assimilate.py
│ ├── WildfireGenerate
│ │ ├── init.py
│ │ ├── models
│ │ │ ├── VAEmodel.py
│ │ │ └── init.py
│ │ ├── task2functions
│ │ │ ├── init.py
│ │ │ ├── feature_extraction.py
│ │ │ ├── predict.py
│ │ │ ├── scoring.py
│ │ │ └── training.py
│ │ └── tests
│ │ ├── init.py
│ │ ├── test_feature_extraction.py
│ │ ├── test_predict.py
│ │ ├── test_scoring.py
│ │ └── test_train.py
│ ├── WildfirePredict
│ │ ├── init.py
│ │ ├── dataset
│ │ │ ├── init.py
│ │ │ ├── dataset.py
│ │ │ └── split_dataset.py
│ │ ├── model
│ │ │ ├── ConvLSTM.py
│ │ │ ├── init.py
│ │ │ └── predict_image.py
│ │ ├── tests
│ │ │ ├── init.py
│ │ │ ├── test_dataset.py
│ │ │ └── test_model.py
│ │ ├── training
│ │ │ ├── init.py
│ │ │ └── training.py
│ │ └── utils
│ │ ├── init.py
│ │ └── plot_sequences.py
│ └── init.py
├── reference.md
├── requirements.txt
├── setup.py
├── task1_handbook.ipynb
├── task2_handbook.ipynb
└── task3_handbook.ipynb

```

---------------------

## Getting started guide 🚀

To set up the environment and install the package, follow these steps:

### Step 1: Create Conda Environment

1. Open the terminal (or Anaconda Prompt for Windows).

2. Run the following command to create a new Conda environment named `wildfire` with Python version 3.8:
   ```sh
   conda create -n wildfire python=3.8
   ```

3. Activate the newly created environment:
   ```sh
   conda activate wildfire
   ```

### Step 2: Install the Package

1. Make sure you are in the directory containing the `setup.py` file.

2. Run the following command to install the package:
   ```sh
   pip install .
   ```

### Step 3: Verify Installation

1. Enter the Python interpreter by typing `python` in the terminal.

2. Try importing your package to ensure it was installed successfully:
   ```python
   import WildfireThomas
   ```

3. If the import completes without errors, your package has been successfully installed and is ready to use.

---

## How to use the package 

Once you've installed the package, you'll be able to run the notebooks in the following order:

1. `task1_handbook.ipynb`
2. `task2_handbook.ipynb`
3. `Choosing_Image.ipynb`
4. `CAE.ipynb`
5. `task3_handbook.ipynb`

------



## ✅ Testing
This project uses pytest for unit testing to ensure code quality and functionality. Regular testing helps in identifying potential issues early and ensures robustness of the codebase.

We aim to maintain high test coverage to ensure the tool's reliability. If you're contributing to the project, please ensure your contirbutions are well-tested, and maintain or improve the current test coverage 😄

-----

## FAQ ❓

- Where can I find the data and the model weights?

There are 2 ways in which you can download the data:

  **First way** Run the first block in the `task1_handbook.ipynb`, manually create a folder named `data` and include everyhting in that folder.

  **Second way** Click on this [google drive link](https://drive.google.com/file/d/1teISZeeps7Mbv-KRQ54kuIx-KdD0yuMe/view?usp=drive_link)
  
----

## 🚧Development Status🚧

Please note that this code is still in active development and might undergo significant changes.

**Model Optimization:** Improving the performance and accuracy of both the RNN and VAE models.

**Scalability:**  Enhancing the models to handle larger datasets and more complex wildfire scenarios.

**User Interface:** Developing a user-friendly interface for interacting with the models and visualizing results.

**Integration:** Integrat additional data sources, such as weather data, to improve prediction accuracy.

----

## License
Distributed under the Apache License. See `license.md` for more information.

---

## Contact
Mok, Leo - leo.mok23@imperial.ac.uk
Guo, Ruida - ruida.guo23@imperial.ac.uk
Veiras Yanes, Jorge - jorge.veiras-yanes23@imperial.ac.uk
Lakatos, Sara - sara.lakatos23@imperial.ac.uk
YU, Chenyang - chenyang.yu23@imperial.ac.uk
Li, Wenxin - wenxin.li23@imperial.ac.uk
Li, CHUHAN - li.li23@imperial.ac.uk
Chadda, Iona Y - iona.chadda23@imperial.ac.uk
