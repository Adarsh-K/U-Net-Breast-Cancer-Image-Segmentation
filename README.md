# Breast Cancer Image Segmentation
### Semantic Segmentation of Triple Negative Breast Cancer(TNBC) Dataset using U-Net CNN architecture

## Triple Negative Breast Cancer
*Triple-negative breast cancer (TNBC) accounts for about 10-15%  of all breast cancers. These cancers tend to be more common in women younger than age 40, who are African-American.*

*Triple-negative breast cancer differs from other types of invasive breast cancer in that they grow and spread faster, have limited treatment options, and a worse prognosis (outcome)*.  - **American Cancer Society**

Thus early stage cancer detection is required to provide proper treatment to the patient and reduce the risk of death due to cancer as detection of these cancer cells at later stages lead to more suffering and increases chances of death. Semantic segmentation of cancer cell images can be used to improvise the analysis and diagonsis of Breast Cancer! Below is such an attempt.

## U-Net
U-Net is a State of the Art CNN architecture for Bio-medical image segmentation. *The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.* It's a Fully Convolutional Network(FCN) therefore it can **work with arbitrary size images!**

<img src="img/U-Net_arch.png">
## Dataset Directory Structure
- train
    * images
        * img
    * label
        * img
- test
    * images
        * img
    * label
        * img
### Sample image from the dataset
![alt-text-1](img/sample_image.png "Original Image") ![alt-text-2](img/sample_label.png "Ground Truth Segmentation Label")


### Sample image from the Canny edge "overlayed" dataset
![alt-text-1](img/sample_image.png "Original Image") ![alt-text-2](img/canny_image.png "Canny Overlayed Image")
