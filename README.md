# Multitask Eye Disease Recognition
Multitask learning for eye disease recognition.  

Work done in Microsoft AI Research. Published in WACV'21. 

Recently, deep learning techniques have been widely used for medical image analysis. While there exists some work on deep learning for ophthalmology, there is little work on multi-disease predictions from retinal fundus images. Also, most of the work is based on small datasets. In this work, given a fundus image, we focus on three tasks related to eye disease prediction: (1) predicting one of the four broad disease categories – diabetic retinopathy, age-related macular degeneration, glaucoma, and melanoma, (2) predicting one of the 320 fine disease sub-categories, (3) generating a textual diagnosis. We model these three tasks under a multi-task learning setup using ResNet, a popular deep convolutional neural network architecture. Our experiments on a large dataset of 40658 images across 3502 patients provides ∼86% accuracy for task 1, ∼67% top-5 accuracy for task 2, and ∼32 BLEU for the diagnosis captioning task.

Link to paper:- https://openaccess.thecvf.com/content/WACV2021/papers/Chelaramani_Multi-Task_Knowledge_Distillation_for_Eye_Disease_Prediction_WACV_2021_paper.pdf

Run the code with:- 
```
python main.py
```

Configuration can be modified in 

```
config.gin
```




