# Intelligent-diagnosis-and-treatment-modeling
2023 postgraduate mathematical modeling

# Project Name

## Overview

This project includes various components aiming to address challenges and tasks in data processing, modeling, and visualization.

## Components

### 1_a

1_a is a program that calculates and establishes a correlation model between volume and time. The data describe the volume of the hematoma and the physical examination time points when hematoma information is acquired.

### 1_b

1_b is a framework composed of three (or more) basic models (traditional machine learning methods) for processing three types of source data. It combines the three output probabilities from these base learners as a new input and inputs them into the final voting machine. The voting machine sets the weights of the learners, computing the weighted probabilities as the final result.

### 2_a
2_a is a soft code attempting to address the regression problem. It employs a polynomial method to fit two-dimensional data types.

### 2_b

2_b is a framework aiming to establish a relationship and transmit latent information from two different data sources. The initial step involves obtaining N clusters based on the first source data, using K-means in this process. According to the results, clustering is expanded to the second source data. The remainder of the code performs straightforward regression work.

### 2_c

2_c details how to plot selected fields.

### 3_a 

3_a is a framework also attempting to handle multi-source data. The main difference from 1_b is that base learners are replaced by deep learning models, and the output of the base learners is a latent feature vector, rather than predicted results.

### 3_b

3_b is a poor example demonstrating how to handle data containing both spontaneous time data and non-time data.

### 3_c

3_c plots various types of charts.

### brain_module.py 
brain_modulerepresents a more advanced task, reconstructing the numerical field data mentioned above into image data.
![brain segment](4.png)
