# Machine Unlearning
# Team name: Air Liquide
Public rank: 57
Private rank: 800

Competition link: https://www.kaggle.com/competitions/neurips-2023-machine-unlearning

This repository is a copy of a private repo hosted in a corporate GitLab.

# Competition Description
This competition, featured in the NeurIPS'23 competition track, focuses on the emerging field of machine unlearning. Participants are tasked with developing algorithms that can remove the influence of a specific subset of training examples, referred to as the "forget set," from a trained machine learning model, without the need for full model retraining. The scenario involves an age predictor trained on face images, and the challenge is to protect the privacy and rights of individuals by unlearning certain training data.

# Our Approaches
We experimented with a variety of techniques from the existing literature, such as SCRUBS and SSD, as well as novel approaches, including noisy gradients, contrastive learning and GANs. To conduct our experiments, we utilized CIFAR10 as a mock dataset, which allowed us to develop and test our unlearning algorithms. Additionally, we pre-trained our own models on CIFAR10, creating an imbalanced version of the dataset to mimic the label distribution found in the actual dataset. This preparation ensured that our unlearning models were well-adapted to real-world scenarios.


## Folders
Each folder contains a different approach, the master folder is **final_framework** that englobes all previous approaches. Moreover, most functions are hosted in utils.py
