# Temporal_CONV and Catroy_Pool for GCN

This project is based on one of our studies for medical image analysis.

We believe Temporal_CONV opens a new way for GNN to process time series information. And Catroy_Pool has a strong generalizability and can be transferred to other medical research fields.

TemporalConv convolves data features independently for each slice
CategoryPool is a test tool for Abnormal hemispherical lateralization

These two modular approaches are an important part of our model Temporal Brain Category Graph Convolutional Network (Temporal-BCGCN)

In this research we have developed a convolution kernel and a pooling kernel based on GCN.

Here we provide the module code for this convolutional pooling kernel

The project follows the Pytorch library at https://pytorch-geometric.readthedocs.io/en/latest/

If you want to run the project directly, it is recommended that you follow the settings of this library

This convolution and pooling kernel does not fit only in our model architecture, so we are publishing the modular source code directly, rather than the full code of the paper, for your better use and learning, but if you are interested, we have published the paper in

Next we will talk about the detailed logic and content of Temporal_CONV and Catroy_Pool in blocks

# Temporal_CONV

![image](https://user-images.githubusercontent.com/33822380/227433968-3cf190c7-cc3b-499e-8a67-42ba5d344264.png)
Fig.1. Comparison of ECConv and TemporalConv convolution processes: (a) Schematic diagram of DSF-BrainNet. (b) Convolution process of applying ECConv to DSF-BrainNet. (c) Convolution process of applying TemporalConv to DSF-BrainNet.

![1679636921124](https://user-images.githubusercontent.com/33822380/227435924-5cec594e-acee-4ea5-94dc-25a84bb5cfaf.png)

![1679637536097](https://user-images.githubusercontent.com/33822380/227437308-0cf7b40f-98ec-4ffa-8a72-d53b4d48195d.png)

# Catroy_Pool

![image](https://user-images.githubusercontent.com/33822380/227434018-82d21020-77c8-4aea-88a5-9e79a54986ce.png)
Fig.2. CategoryPool Process Diagram

![1679637388430](https://user-images.githubusercontent.com/33822380/227436716-1529a172-2d62-42e0-8aee-dfaee9d2e8f0.png)

![1679637349821](https://user-images.githubusercontent.com/33822380/227436678-c758b86d-d97d-4333-ae28-e067fd039195.png)

