# Machine Learning in Data Streams

**Project name: Real-time CTR prediction with online feature selection**

This repository contains the code, data analysis, and documentation for the project developed for the *Machine Learning Methods for Data Streams* course. 

The primary objective of this project is to address concept drift and feature drift in highly dimensional, sparse, and imbalanced data streams. We propose a novel extension to the streaming random patches (SRP) algorithm by integrating dynamic online feature selection (OFS) to minimize the adaptation gap ($t_{recovery}$) and reduce Log Loss.

# Datasets downloaded from Kaggle:
1. [Avazu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction/data) - A large-scale dataset containing click-through data for mobile ads, with 10 million instances and 22 categorical features.
2. [Criteo Display Advertising Challenge](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset) - A dataset with 45 million instances and 13 numerical and 26 categorical features, used for predicting click-through rates in display advertising.

Both of datasets that need to be downloaded are train sets. 

## Repository structure

* `avazu/` - Scripts and data processing for the Avazu dataset.
* `criteo/` - Scripts and data processing for the Criteo dataset.
* `stream-ctr/` - Algorithmic implementation in Java, integrating OFS into the Massive Online Analysis (MOA) framework.
* `latex/` - LaTeX source code for project report and presentation.
* `drift_detection_avazu.py` and `drift_detection_criteo.py` - Python scripts using the `river` library (ADWIN) to verify concept drift in real-world streams empirically.
* `adaptation_gap.py` - Script to visualize the adaptation gap and error spikes from the synthetic Agrawal generator.
* `evaluation_plots.py` - Visualization tools for prequential evaluation results.
* `dumpFile.csv` and `predictions_agrawal.pred` - MOA output files used for baseline performance tracking.
* `requirements.txt` - Python dependencies.

## Prerequisites
Make sure you have Python 3.8+ installed. You will also need Java to run the MOA framework components.

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## How to run the code and reproduce the results:
1. Download the Avazu and Criteo datasets from Kaggle and place them in the respective `avazu/` and `criteo/` directories.
2. Run the avazu_download.py and criteo_download.py scripts to preprocess the datasets.
3. Run the feature_hashing.py scripts for criteo and avazu to perform feature hashing.
4. Run MLDataStreams\stream-ctr\src\main\java\stream\Main.java file to execute the proposed algorithm and baseline evaluations in MOA.
5. Use the evaluation_plots.py script to visualize the results from MOA output files.

## Contributors
* **Aleksandra Kłos** - EDA, data preprocessing, baseline evaluation 
* **Hubert Jaczyński** - Analytics, final evaluation, fixes in algorithm implementation
* **Jakub Oganowski** - Algorithm implementation in Java

### Each of us has been somehow involved in the concept drift detection. 
