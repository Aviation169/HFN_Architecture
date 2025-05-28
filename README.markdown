# HybridFeatureNet: A Lightweight CNN for CIFAR-10 with ASCR Layer

## Overview
Welcome to **HybridFeatureNet (HFN)**! This is a custom convolutional neural network (CNN) designed for the CIFAR-10 dataset, achieving a test accuracy of **90.83%** with only **2,127,121 parameters**. HFN incorporates innovative components like the **ASCR Layer** (Adaptive Suppression and Channel Refinement Layer), multi-scale feature extraction, attention mechanisms, and global context modules, making it a lightweight yet effective model for image classification.

![HFN Architecture](Architecture_images/hfn_architecture.png)

This project is shared for **experimental and educational purposes**, aimed at inspiring young developers like you to explore deep learning, create new architectures, or experiment with the ASCR Layer I introduced. While HFN may not compete with state-of-the-art models (e.g., ViT with 99% accuracy), its efficiency and custom design make it a great starting point for learning and innovation.

## Features
- **Lightweight Architecture**: Only 2.13M parameters, making it much smaller than models like ResNet-50 (25.6M) or VGG-16 (15–20M for CIFAR-10).
- **Solid Performance**: Achieves 90.83% accuracy on the CIFAR-10 test set.
- **Innovative ASCR Layer**: A custom layer for adaptive feature suppression and channel refinement, which you can experiment with or adapt to other models.
  
  ![ASCR Layer](Architecture_images/ascr_layer_neural_network.png)
- **Multi-Scale and Attention Mechanisms**: Incorporates multi-scale feature extraction, attention modules, and global context for improved feature learning.
- **Easy to Experiment With**: Well-commented code, modular design, and a simple training pipeline for you to modify and build upon.
  
  ![HFN Neural Image](Architecture_images/hfn_neural_network.png)

## Project Structure
```
hybrid_feature_net/
│
├── Architecture.py               # Main script with model, training, and evaluation
├── best_hybrid_feature_net.pth   # Model weight
├── README.md                     # Project documentation (you're reading it!)
├── requirements.txt              # Dependencies
└── LICENSE                       # MIT License file
```

**Note**: This Architecture is just for testing and experimental purpose

## Installation
Follow these steps to set up and run the project on your local machine.

### Prerequisites
- Python 3.7 or higher
- PyTorch with CUDA support (if using a GPU; CPU works too)
- A basic understanding of deep learning and Python

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Aviation169/HFN_Architecture.git
   cd Architecture
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   - `torch`
   - `torchvision`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`
   - `pillow`

3. **Verify Installation**:
   Ensure PyTorch is installed correctly:
   ```python
   python -c "import torch; print(torch.__version__)"
   ```
   If you have a GPU, check CUDA availability:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage
The `Architecture.ipynb` script includes everything you need to train, save, and evaluate the model on CIFAR-10, as well as test it on an unseen image.

### 1. Train the Model
- **Run the Script**:
  ```bash
  Architecture.ipynb
  ```
- **What Happens**:
  - Downloads the CIFAR-10 dataset to the `./data` directory.
  - Trains the `HybridFeatureNet` model with early stopping (patience=15 epochs).
  - Saves the best model weights as `best_hybrid_feature_net.pth` based on the lowest test loss.
  - Generates a confusion matrix for the test set and (optionally) tests an unseen image.

- **Training Details**:
  - Optimizer: Adam (learning rate=0.001, weight decay=5e-5)
  - Scheduler: Cosine Annealing LR (T_max=100)
  - Data Augmentation: Random crop, horizontal flip, color jitter, random erasing
  - Batch Size: 128
  - Max Epochs: 100 (or until early stopping)

### 2. Evaluate and Test
- **Confusion Matrix**: After training, the script automatically generates a confusion matrix for the CIFAR-10 test set (10,000 images).
- **Test on an Unseen Image**:
  - Update the `image_path` variable in `Architecture.ipynb`:
    ```python
    image_path = 'path/to/your/truck_image.jpg'  # Replace with your image path
    ```
  - Run the script to see the predicted class, class probabilities, and a bar chart of probabilities.
  - Example output:
    ```
    Predicted class: truck
    Class probabilities:
    airplane: 2.34%
    automobile: 10.12%
    bird: 1.67%
    cat: 0.43%
    deer: 0.21%
    dog: 0.78%
    frog: 0.56%
    horse: 0.89%
    ship: 3.45%
    truck: 80.55%
    ```
    A bar chart will also display, with the predicted class highlighted in orange.

### 3. Modify and Experiment
- **Add New Layers**: Try integrating the ASCR Layer into other architectures or modify its parameters (e.g., `reduction_ratio`, `suppression_prob`).
- **Change Architecture**: Adjust the number of stages, filters, or add new modules to `HybridFeatureNet`.
- **Hyperparameters**: Experiment with learning rate, batch size, or data augmentation techniques.

## Model Architecture
`HybridFeatureNet` is designed to be lightweight and effective, with the following components:
- **Stem**: ConvBlock (3 → 32, kernel=3x3)
- **Stage 1**:
  - MultiScaleFeatureExtractor (32 → 64): Extracts features at multiple scales (1x1, 3x3, 5x5, pooling).
  - AttentionModule (64): Channel and spatial attention.
  - ConvBlock (64 → 64, kernel=3x3)
- **Stage 2**:
  - MultiScaleFeatureExtractor (64 → 128)
  - AttentionModule (128)
  - ASCR Layer (128): Custom layer for adaptive feature suppression.
  - Dropout (0.3)
  - ConvBlock (128 → 128, kernel=3x3)
- **Stage 3**:
  - MultiScaleFeatureExtractor (128 → 256)
  - GlobalContextModule (256): Captures global dependencies.
  - ConvBlock (256 → 256, kernel=3x3)
- **Output**: Global average pooling, fully connected layer (256 → 10)

### ASCR Layer
The **Adaptive Suppression and Channel Refinement (ASCR) Layer** is a custom contribution:
- Combines channel and spatial attention with adaptive suppression.
- Key Parameters:
  - `reduction_ratio`: Controls channel reduction (default=8).
  - `suppression_prob`: Probability of suppressing low-weight features (default=0.2).
  - `temperature`: Learnable parameter for attention scaling (default=1.0).
- Purpose: Enhances important features while suppressing less relevant ones, improving model focus.

## Performance
- **Accuracy**: 90.83% on CIFAR-10 test set (10,000 images).
- **Parameter Count**: 2,127,121—much smaller than models like ResNet-50 (25.6M), DenseNet201 (20M), or VGG-16 (15–20M for CIFAR-10).
- **Strengths**:
  - Efficient design with competitive accuracy.
  - Performs well on classes like automobiles (95.4% correct), trucks (95.0%), and ships (93.9%).
- **Weaknesses**:
  - Struggles with cats (80.5% correct) and dogs (87.0%) due to visual similarity.
  - Below state-of-the-art models like ViT (99%) or Airbench (94–96%).

### Confusion Matrix
Below is the confusion matrix for the CIFAR-10 test set, showing true labels (rows) vs. predicted labels (columns):

| True ↓ / Predicted → | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|----------------------|----------|------------|------|-----|------|-----|------|-------|------|-------|
| airplane             | 928      | 4          | 16   | 6   | 10   | 1   | 2    | 4     | 20   | 9     |
| automobile           | 7        | 954        | 0    | 0   | 0    | 0   | 2    | 1     | 8    | 28    |
| bird                 | 19       | 2          | 881  | 29  | 27   | 14  | 8    | 1     | 1    | 2     |
| cat                  | 5        | 4          | 29   | 805 | 38   | 86  | 15   | 9     | 1    | 8     |
| deer                 | 4        | 1          | 26   | 23  | 903  | 16  | 13   | 12    | 2    | 0     |
| dog                  | 3        | 0          | 11   | 79  | 15   | 870 | 7    | 13    | 0    | 2     |
| frog                 | 3        | 0          | 20   | 31  | 12   | 9   | 922  | 2     | 0    | 1     |
| horse                | 7        | 0          | 9    | 15  | 16   | 20  | 1    | 931   | 0    | 1     |
| ship                 | 29       | 5          | 5    | 7   | 2    | 0   | 2    | 0     | 939  | 11    |
| truck                | 7        | 27         | 2    | 2   | 1    | 1   | 2    | 2     | 6    | 950   |

- **Overall Accuracy**: (928 + 954 + 881 + 805 + 903 + 870 + 922 + 931 + 939 + 950) / 10000 = 90.83%

## Contributing
This project is all about experimentation and learning! Here’s how you can get involved:
- **Fork and Experiment**: Fork the repository, try new architectures, or tweak the ASCR Layer.
- **Add Features**: Implement new layers, improve data augmentation, or optimize training.
- **Share Your Results**: Submit a pull request with your changes, or open an issue to discuss your ideas.
- **Ask Questions**: If you're stuck or need guidance, open an issue—I’d love to help!

### Ideas for Experimentation
- **Modify ASCR Layer**: Adjust `suppression_prob` or `reduction_ratio`, or apply it to other stages.
- **Add Depth**: Increase the number of stages or filters to boost accuracy.
- **Try Other Datasets**: Test HFN on CIFAR-100 or tiny ImageNet.
- **Optimize Training**: Experiment with learning rate schedules, optimizers, or test-time augmentation.

## License
This project is licensed under the MIT License—see the License file for details. Feel free to use, modify, and share this code as you wish!

## Acknowledgements
- **Built With**: PyTorch, torchvision, NumPy, Matplotlib, Seaborn, Scikit-learn, Pillow.
- **Dataset**: CIFAR-10 by Alex Krizhevsky.
- **Inspiration**: Modern CNN architectures like ResNet, DenseNet, and Vision Transformers.
- **Message to Young Developers**: This project started as an experiment, and you can do the same! Don’t be afraid to try new ideas, create your own layers, or build unique models. The deep learning community grows when we share and learn together—keep innovating! 🚀

## Contact
If you have questions, ideas, or just want to chat about deep learning, feel free to reach out:
- Email: [akajay14955j@gmail.com]

Happy coding, and let’s keep pushing the boundaries of what’s possible with deep learning! 🎉
