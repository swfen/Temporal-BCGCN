# Temporal_CONV and Catroy_Pool

Temporal_CONV and Catroy_Pool for GCN

This project is based on one of our studies for medical image analysis

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

# Catroy_Pool

![image](https://user-images.githubusercontent.com/33822380/227434018-82d21020-77c8-4aea-88a5-9e79a54986ce.png)
Fig.2. CategoryPool Process Diagram

