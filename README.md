# Convolutional-Recurrent Architectures for Artwork Classification

## Abstract
This project aims to classify artwork images into various attributes (Style, Genre, and Artist) using convolutional-recurrent architectures. pre-trained CNN based model was combined with LSTM layers to capture both spatial and sequential patterns. This pipeline handles data cleaning, label encoding, training, evaluation metrics calculation, and outlier detection. This work is compare two different backbone models (ResNet and EfficientNet) to assess their performances on the WikiArt dataset.

---

## Approach
1. **Preprocessing & Label Encoding**: We merge the WikiArt dataset splits (train/validation) to ensure consistent label encoding. Invalid images are removed.
2. **Model Architectures**:   - 
   - **ResNet + LSTM**: Similarly, we remove the final FC layers from a ResNet (e.g., ResNet50), then pool and feed the features into LSTM layers.
   - **EfficientNetB0 + LSTM**: EfficientNetB0 is used as the feature extractor, followed by pooling and an LSTM layer.
3. **Training**: We train each model on the cleaned dataset. We also incorporate a `WeightedRandomSampler` for minority classes, ensuring more balanced training.
4. **Outlier Detection**: After training, we run inference on the test set and mark samples as potential outliers when the confidence for their “true” label is below a set threshold (e.g., 30%).

---

## Dataset
- We use the [WikiArt dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md), which contains images labeled by artist, style, and genre. 
- The dataset is split into train/validation partitions, merged into a single pool, and we manually create a 70/15/15 train/validation/test split for experimentation.
- We visualize label distributions (genre, style, artist) to understand class imbalances. Bar plots and confusion matrices are used to see how many samples exist for each class.

**Sample Distributions**:
- We generate bar charts of the frequency of styles, genres, and artists.
- We remove any rows pointing to missing or invalid image files.
<img width="347" alt="data-dist" src="https://github.com/user-attachments/assets/d63c611d-e114-4dc4-8864-63ee778be4c2" />
<img width="347" alt="data-dis" src="https://github.com/user-attachments/assets/80472e89-1cd4-440e-882e-2274a26be257" />

---

## Model 
**ResNet + LSTM Architecture**

1. A **ResNet50** backbone (pretrained on ImageNet) is truncated before its global average pool and FC layers.
2. The feature map is **pooled adaptively** to a fixed size (7×7), creating a sequence of spatial features.
3. An **LSTM** processes these sequential features. The final hidden state is used for three separate classification heads:
   - Style Classification
   - Genre Classification
   - Artist Classification

Each head is a simple linear layer projecting from the LSTM’s hidden state to the respective number of classes.


---

**EfficientNetB0 + LSTM Architecture**

- **EfficientNetB0 Backbone**:
  - Utilizes EfficientNetB0 pretrained on ImageNet to extract feature representations.
  - The classifier head is removed, retaining only convolutional feature layers which produce a rich spatial feature map with 1280 feature channels.

- **Adaptive Pooling**:
  - Features extracted from EfficientNet are passed through an adaptive average pooling layer, standardizing spatial feature dimensions to a fixed size.


- **LSTM Layer**:
  - A single-layer Long Short-Term Memory (LSTM) network processes these sequences to capture spatial dependencies as sequential information.
  - The final hidden state from the LSTM, which encapsulates learned sequential context, serves as the input to subsequent classification heads.

- **Classification Heads**:
  - Three distinct linear classifier heads, each corresponding to one attribute of artwork (style, genre, artist), receive the LSTM’s final hidden state.
  - Each head outputs logits predicting class probabilities for their respective categories.


---

## Evaluation Metrics & Outliers
1. **Metrics**:
   - **Accuracy**: Percentage of correctly predicted samples.
   - **Confusion Matrix**: For interpretability, especially focusing on top classes (e.g., top 10 most frequent styles), we display confusion matrices to identify misclassifications.

2. **Outlier Detection**:
   - After obtaining predictions, we compare the model’s probability (softmax) of the “true” label. If that probability is below a threshold (e.g., 0.3), we flag the image as a potential outlier.
   - We then display a small grid of outlier images with their assigned genre/style/artist and confidence scores.

---

## Results Analysis
We trained and evaluated two backbones (ResNet, EfficientNet) followed by an LSTM, accurately representing the final test accuracy metrics based on the images in test dataset:

### Results Analysis: Comparison of ResNet+LSTM vs EfficientNet+LSTM

We conducted experiments comparing two convolutional-recurrent architectures: ResNet50 combined with LSTM and EfficientNetB0 combined with LSTM. Below are their comparative performances:

#### 1. **ResNet50 + LSTM:**
- **Final Test Accuracy:**
  - **Style:** 36.35%
  - **Genre:** 29.79%
  - **Artist:** 19.50%
<img width="499" alt="resnet_accuracy" src="https://github.com/user-attachments/assets/fa5d9e8c-44f0-482e-92d1-e32b39d5151c" />

The training and validation loss curves demonstrate gradual convergence, although validation loss remained relatively high and exhibited instability towards the later epochs. The validation accuracy plots show slow improvements across epochs, particularly for genre and artist predictions. Style classification appeared to be consistently higher in accuracy compared to the other two categories.

#### 2. **EfficientNetB0 + LSTM:**
- **Final Test Accuracy:**
  - **Style:** 47.34%
  - **Genre:** 35.76%
  - **Artist:** 32.57%
<img width="499" alt="efficientnet_accuracy " src="https://github.com/user-attachments/assets/600adfd8-be04-4ff2-b5a7-5059aff44945" />

EfficientNetB0 clearly outperformed ResNet50 in terms of final accuracy across all three categories: style, genre, and artist. The training and validation losses show clearer convergence with less fluctuation compared to ResNet50. EfficientNetB0 also exhibited higher and more stable validation accuracies, suggesting better generalization and robustness on unseen data.


### conclusion of Comparison:
- EfficientNetB0+LSTM significantly surpassed ResNet50+LSTM on the WikiArt classification tasks.
- **Style Accuracy Improvement:** approximately +10.99% points (47.34% vs 36.35%).
- **Genre Accuracy Improvement:** approximately +5.97% points (35.76% vs 29.79%).
- **Artist Accuracy Improvement:** approximately +13.07% points (32.57% vs 19.50%).

EfficientNetB0’s better performance might be attributed to its architecture designed explicitly for efficiency and effectiveness with relatively smaller datasets and input sizes. The deeper and broader feature extraction capabilities of EfficientNetB0 likely resulted in capturing more relevant artistic features and nuances.

---
## Confusion Matrix (Top 10 Styles)
Below are the **style confusion matrices** on the **test set**, focusing on the **top 10 most frequent styles**,  comparing for **ResNet+LSTM** and **EfficientNet+LSTM**:

- **ResNet+LSTM **:  

<img width="415" alt="confusion_resnet" src="https://github.com/user-attachments/assets/27f12f6e-ccc6-423b-859e-7f0201d88d61" />


- **EfficientNet+LSTM **:  
<img width="415" alt="confusion_efficient" src="https://github.com/user-attachments/assets/d8744b7c-e89d-4566-8c97-679590f7c242" />


In both matrices, misclassifications occur most frequently between stylistically similar groups (e.g., **Impressionism** ↔ **Realism**, **Expressionism** ↔ **Post-Impressionism**). These results highlight the inherent difficulty in distinguishing certain artistic styles purely from image features, as many transitional or borderline works share overlapping visual cues.

---

## Outlier Detection
After evaluating on the test set, we flag any sample whose **confidence in the “true” label** (i.e., the softmax probability assigned to the ground-truth style/genre/artist) falls below a **0.3 threshold**. These flagged samples are potential outliers—either genuinely **atypical** artworks or possibly **mislabeled** items.

- **ResNet+LSTM**: 
  - Detected **1,636 potential outliers** in the test set.  
  - The *top 10 outliers* (lowest confidence) often belong to borderline or ambiguous stylistic categories.
<img width="713" alt="out_resnet" src="https://github.com/user-attachments/assets/db7ba4ce-602b-4dbf-8ba2-670478ced551" />

- **EfficientNet+LSTM**:  
  - Detected **1,485 potential outliers** in the test set.  
  - Similarly, the *top 10 outliers* typically exhibit indistinct features or near-equal probabilities across multiple styles/genres/artists.
<img width="698" alt="out_efficient" src="https://github.com/user-attachments/assets/553d27f9-a50f-4328-af1e-64a4207bcff5" />

We list a few examples of these low-confidence predictions in each case. While outlier detection does not definitively confirm a mislabel, it can help **focus manual review** on suspicious samples.

---


## Possible Improvements
1. **Longer Training / Hyperparameter Tuning**: More epochs, fine-tuning learning rates, or deeper LSTM layers might improve performance.
2. **Data Augmentation**: More sophisticated augmentations (color jitter, random rotations) could help the model generalize.
3. **Class-Aware Sampling**: Instead of weighting only by genre, expand to also consider style or artist imbalance.
4. **Advanced Architectures**: Other state-of-the-art networks (e.g., Vision Transformers) might capture style/genre nuances better.
5. **Multi-Task Loss Tweaking**: Adjust the weight of style vs. genre vs. artist losses if one task is more crucial or more difficult.

---

## Conclusion
In summary:

1. **EfficientNet+LSTM** consistently outperforms **ResNet+LSTM** across **style**, **genre**, and **artist** classification tasks on WikiArt, showing higher accuracies and more stable training curves.

2. The **confusion matrices** for **style (top 10)** reveal recurring challenges in distinguishing closely related styles. **Romanticism** emerged with the highest diagonal counts for both models, though other styles (e.g., **Impressionism**, **Expressionism**) frequently overlapped.

3. **Outlier detection** flagged images with low confidence for their true labels. Some of these appear visually ambiguous or potentially mislabeled.

Overall, **EfficientNet+LSTM** offers a more robust feature representation for complex art classification. Future work could explore **longer training**, **data augmentation**, or **alternative backbones** (e.g., Vision Transformers) to further increase accuracy and reduce confusion among similar styles.

Overall, EfficientNetB0+LSTM demonstrated a clear advantage for this classification task and would be the recommended model of the two evaluated architectures.

---

## Implementation Guide

This section details how to reproduce the experiments presented in this repository. The code is organized into two .pynb (upload them into Kaggle notebooks) —one for **ResNet + LSTM** and one for **EfficientNet + LSTM**. Both notebooks utilize the WikiArt dataset hosted on Kaggle, so you will need access to Kaggle notebook and the WikiArt dataset to follow along precisely.

### 1. Kaggle Environment Setup

1. **Create a Kaggle account** if you do not already have one.  
2. **Fork/Copy** the notebooks:
   - resnet-lstm.ipynb
   - efficientnet-lstm.ipynb

3. **Add the WikiArt dataset** to your Kaggle environment:
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
   - Search for the WikiArt dataset used here, or upload the dataset to your private Kaggle workspace if necessary.
   - In your Kaggle Notebook settings, add the dataset under **Add a Dataset**.

4. **Enable GPU Runtime**:
   - In the Kaggle Notebook settings, select **Accelerator = GPU** for faster training, as these models are computationally intensive.

### 2. Directory Structure

When working on Kaggle, notebooks and datasets are placed in the following structure by default:
```
/kaggle/
  |-- input/
       |-- wikiart-dataset/  (Your WikiArt dataset here)
  |-- working/
       |-- resnet-lstm.ipynb
       |-- efficientnet-lstm.ipynb       
```
Depending on your setup, you might need to adjust file paths in the notebooks if your folder names differ.

### 3. Running the Notebooks

#### A. `resnet-lstm.ipynb`
1. **Import and Install Requirements**  
   Ensure all required libraries (PyTorch, torchvision, etc.) are installed. Kaggle notebooks usually come with these preinstalled, but run `!pip install ...` if needed.

2. **Data Loading & Cleaning**  
   - The notebook merges the WikiArt dataset splits (train, validation) if they are in separate folders.  
   - It performs label encoding for style, genre, and artist.  
   - It filters out invalid or missing images.

3. **Model Definition: ResNet + LSTM**  
   - Loads a pretrained ResNet (e.g., ResNet50) from `torchvision.models`.  
   - Removes the final fully connected layers and replaces them with an LSTM-based sequence processor and classification heads for style, genre, and artist.

4. **Training**  
   - The notebook trains the model on the preprocessed WikiArt images.  
   - Weights are saved periodically or after the final epoch (depending on your configuration).

5. **Evaluation & Confusion Matrix**  
   - Generates accuracy and weighted F1/Recall metrics.  
   - Plots confusion matrices (if using `matplotlib` or `seaborn`).  
   - Performs outlier detection (low-confidence predictions) and saves flagged images.

#### B. `efficientnet-lstm.ipynb`
1. **Import and Install Requirements**  
   Similar environment setup as the ResNet notebook. EfficientNet support is found in `torchvision.models` (PyTorch versions 1.12+).

2. **Data Preparation**  
   - Loads the same WikiArt data, merges splits, cleans invalid samples, and encodes labels.

3. **Model Definition: EfficientNet + LSTM**  
   - Initializes EfficientNetB0 (pretrained on ImageNet).  
   - Removes the default classifier head.  
   - Feeds adaptive-pooled features into an LSTM layer, then attaches separate classification heads for style, genre, and artist.

4. **Training**  
   - Similar procedure to ResNet: sets up an optimizer (e.g., Adam or SGD), a loss function (cross-entropy for each classification head), and uses a `WeightedRandomSampler` for imbalanced classes.

5. **Evaluation & Outlier Detection**  
   - Tracks the same set of metrics: accuracy, weighted F1, and recall.  
   - Computes confusion matrices for the top 10 most frequent classes.  
   - Flags outliers where the ground-truth class probability is below 0.3.

### 4. Modifying Hyperparameters

Both notebooks include configuration cells where you can change:
- **Epochs** (default ~10–20 in examples).  
- **Batch Size** (depending on GPU memory).  
- **Learning Rate** (standard ~1e-4 or 1e-3 range, can be tuned).  
- **Scheduler** (optional) to adjust the learning rate during training.  
- **LSTM settings** such as hidden dimensionality and number of layers.

### 5. Monitoring Training
- Use the Kaggle output cells to watch **training/validation loss** and **accuracy** logs.  
- Generate and review **tensorboard**-like plots if desired (some code cells show how to create line plots of epoch metrics).

### 6. Saving & Exporting Results
- Both notebooks provide cells to **save the trained models** (e.g., `model.pth`) to `/kaggle/working/`.  
- You can also download these files locally or share them within your Kaggle environment.  
- You may export the entire notebook as `.ipynb` or `.html` for external viewing.

### 7. Potential Issues & Troubleshooting
- **Memory Constraints**: Large batch sizes or high-resolution images may exceed GPU memory in free Kaggle environments. Reduce batch size or image size accordingly.  
- **Missing/Corrupt Images**: If your WikiArt dataset has different folder structures, double-check file paths in the data-loading cells.  
- **PyTorch Version**: Ensure you have a PyTorch version that supports `torchvision.models.efficientnet_b0` (PyTorch 1.12+ is recommended).

### 8. Extending the Notebooks
- **Custom Backbones**: Feel free to swap ResNet50/EfficientNetB0 with other pretrained models.  
- **Additional Data Augmentations**: Add advanced augmentations (e.g., cutout, random erasing) to the transform pipeline.  
- **Multi-Task Weights**: Adjust loss weighting if one classification task (e.g., artist) is more important.  
- **Fine-Tuning**: Unfreeze more layers of the CNN backbone to allow deeper fine-tuning for improved performance.

---

This end-to-end guide should help you **replicate, modify, and extend** the convolutional-recurrent (CNN+LSTM) pipelines for artwork classification in Kaggle notebooks. By following the steps above, you can train and evaluate the models on your own subset of the WikiArt dataset, generate confusion matrices, and detect potential outlier samples based on low-confidence predictions.
