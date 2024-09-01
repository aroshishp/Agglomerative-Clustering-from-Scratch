# Agglomerative Clustering from Scratch

This directory contains the code for implementaation of agglomerative clustering from scratch to a dataset that simulates an evolving ecosystem of animals.

The clustering is implemented in the following way:
* The dataset is preprocessed and unnecessary columns like 'Species Name' are dropped.
* The categorical columns like 'Diet' and 'Habitat' are one-hot encoded.
* The proximity matrix is using Euclidean Distances between the row vectors of the dataframe.
* The individual points are converted to clusters containing themselves.
* The smallest distance is identified and the corresponding points are clustered into one.
* Between multiple clusters, centroid linkage is used to be less computationally intensive.
* The clustering is repeated till there is a single cluster left.
* Based on the clustering, a dendrogram is created and plotted.
* To better see the evolution of the animals over time steps, an interactive tool is created using streamlit where the user can manually adjust the time step using a slider and also add new animals or mutate existing animals.

A description of the files in this directory is given below:
1. [Epoch_Session_3_LR.pdf](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/Epoch_Session_3_LR.pdf) - Learning Report (difference between agglomerative and divisive clutering, code outline, checkpoint questions)
2. [adap-ecosys-dataset.csv](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/adap-ecosys-dataset.csv) - Dataset
3. [proximity.ipynb](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/proximity.ipynb) - Code to calculate and save proximity matrix
4. [output.csv](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/output.csv) - 1000 by 1000 proximity matrix for TimeStep 0
5. [clustering.ipynb](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/clustering.ipynb) - Code to implement agglomerative clustering for submatrices of output.csv
6. [all_time_steps.ipynb](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/all_time_steps.ipynb) - Code to loop through all timesteps and plot dendrogram.
7. [interactive.py](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/interactive.py) - Streamlit Tool Code

To run the streamlit tool, download [interactive.py](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/interactive.py) and [adap-ecosys-dataset.csv](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/adap-ecosys-dataset.csv). If not installed, download streamlit using:
```
pip install streamlit
```
Then run:
```
streamlit run interactive.py
```

A demo is provided below.

![Demo](https://github.com/IITH-Epoch/Learning_phase_2024-25/blob/main/Aroshish/Session%203/Agglomerative%20Clustering%20Demo.gif)
