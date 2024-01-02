# GLocal-K: Global and Local Kernels for Recommender Systems
Paper: [GLocal-K: Global and Local Kernels for Recommender Systems](https://arxiv.org/pdf/2108.12184.pdf).

![GLocal_K_overview](https://user-images.githubusercontent.com/41948621/131093771-39d86126-6be6-4fc8-bcda-3eab8fd2c181.png)

## 1. Introduction
The proposed matrix completion framework based on global and local kernels, called GLocal-K, includes two stages: 1) pre-training an autoencoder using the local kernelised weight matrix, and 2) fine-tuning the pre-trained auto encoder with the rating matrix, produced by the global convolutional kernel. This repo provide the benmark with the processed data from [Movie Recommender Systems](https://www.kaggle.com/code/rounakbanik/movie-recommender-systems/input).

## 2. Setup
Download this repository. As the code format is .ipynb, there are no settings but the Jupyter notebook with GPU.

## 4. Run
1. Use the data in dir `processed_data_for_matrix_completion` or create a csv file with the same format.
2. Insert the data path in the main code.
3. Run the notebook and see the result.

## 3. Requirements
* numpy
* scipy
* torch