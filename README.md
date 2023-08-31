# Temporal_CONV and Catroy_Pool for GCN

This project is based on one of our studies for medical image analysis. Specifically, this is a binary classification study of people with schizophrenia versus normal people

We believe Temporal_CONV opens a new way for GNN to process time series information. And Catroy_Pool has a strong generalizability and can be transferred to other medical research fields.

TemporalConv convolves data features independently for each slice

CategoryPool is a test tool for Abnormal hemispherical lateralization

These two modular approaches are an important part of our model Temporal Brain Category Graph Convolutional Network (Temporal-BCGCN)

In this research we have developed a convolution kernel and a pooling kernel based on GCN. This convolution and pooling kernel does not fit only in our model architecture, so we are publishing the modular source code directly, rather than the full code of the paper. 
for your better use and learning, please read the paper in https://arxiv.org/abs/2304.01347

The project follows the Pytorch library at https://pytorch-geometric.readthedocs.io/en/latest/

If you want to run the project directly, it is recommended that you follow the settings of this library
