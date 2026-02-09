# Privacy-Aware Person Re-Identification

A comprehensive implementation and analysis of privacy-preserving techniques for Person Re-Identification (Re-ID) systems using Differential Privacy. This project demonstrates the privacy-utility trade-off in deep learning models trained on the Market-1501 dataset.

## Overview

Person Re-Identification is the task of matching pedestrian images across different camera views. While highly useful for surveillance and security applications, Re-ID systems raise significant privacy concerns as they can be vulnerable to:

- **Membership Inference Attacks**: Determining if a person's image was part of the training data
- **Attribute Inference Attacks**: Inferring sensitive attributes about individuals

This project implements a privacy-aware Re-ID system using **Differential Privacy (DP)** to mitigate these risks while maintaining acceptable utility for person re-identification tasks.

## Features

- **Baseline Re-ID Model**: Standard ResNet50-based person re-identification model
- **Differential Privacy Training**: DP-SGD implementation with configurable privacy budgets (ε)
- **Privacy Attacks**:
  - Membership inference attack evaluation
  - Attribute inference attack evaluation
- **Privacy-Utility Analysis**: Comprehensive trade-off analysis across different privacy budgets
- **Visualization**: Automated generation of privacy-utility trade-off plots

## Architecture

### Model Architecture
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Embedding Layer**: 2048 → 512 dimensions
- **Loss Functions**:
  - Cross-Entropy Loss for classification
  - Triplet Loss for metric learning

### Privacy Mechanism
- **Algorithm**: DP-SGD (Differentially Private Stochastic Gradient Descent)
- **Privacy Accounting**: Rényi Differential Privacy
- **Configurable Parameters**:
  - Privacy budget (ε): 1.0, 3.0, 5.0, 8.0, 10.0, 50.0
  - Delta (δ): 1e-5
  - Gradient clipping norm: 1.0

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dhanjo/Privacy-Aware.git
cd Privacy-Aware
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

This project uses the **Market-1501** dataset for person re-identification.

### Manual Download

1. Visit the official Market-1501 dataset page:
   - **Official page**: https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
   - **Alternative**: Search for "Market-1501 dataset download" for mirror links

2. Download `Market-1501-v15.09.15.zip` (approximately 330 MB)

3. Extract the dataset to the `data` directory:
```bash
# Create data directory
mkdir -p data

# Extract the downloaded zip file
unzip Market-1501-v15.09.15.zip -d data/
```

4. Verify the directory structure:
```
data/Market-1501/
├── bounding_box_train/    # Training images (~12,936 images, 751 identities)
├── bounding_box_test/     # Gallery images (~19,732 images, 750 identities)
├── query/                 # Query images (~3,368 images)
└── gt_bbox/               # Ground truth bounding boxes
```

### Dataset Statistics
- **Training set**: 12,936 images of 751 identities
- **Test set**: 19,732 images of 750 identities
- **Query set**: 3,368 images
- **Image size**: 128x64 pixels (variable)
- **Cameras**: 6 cameras

## Usage

### Quick Start - Full Pipeline

Run the complete analysis pipeline:

```bash
python privacy_utility_tradeoff.py
```

This will:
1. Train a baseline model (no privacy protection)
2. Train multiple DP models with different privacy budgets
3. Evaluate Re-ID performance (mAP, Rank-1, Rank-5)
4. Perform membership inference attacks
5. Perform attribute inference attacks
6. Generate privacy-utility trade-off visualizations

### Individual Components

#### 1. Train Baseline Model
```bash
python train_baseline.py
```

Options:
- `--batch-size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 0.0003)
- `--epochs`: Number of training epochs (default: 30)

#### 2. Train DP Model
```bash
python train_dp.py --epsilon 5.0
```

Options:
- `--epsilon`: Privacy budget (smaller = more privacy, default: 5.0)
- `--delta`: Privacy parameter (default: 1e-5)
- `--max-grad-norm`: Gradient clipping threshold (default: 1.0)
- `--batch-size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 15)

#### 3. Extract Embeddings
```bash
python extract_embeddings.py --model-path results/baseline_model.pth --model-name baseline
```

#### 4. Evaluate Re-ID Performance
```bash
python evaluate_reid.py --model-name baseline
```

#### 5. Membership Inference Attack
```bash
python attack_membership.py --model-path results/baseline_model.pth --model-name baseline
```

#### 6. Attribute Inference Attack
```bash
python attack_attribute.py --model-path results/baseline_model.pth --model-name baseline
```

## Results

All results are saved in the `results/` directory:

```
results/
├── baseline_model.pth                    # Trained baseline model
├── dp_model_eps_*.pth                    # DP models for each epsilon
├── baseline_reid_metrics.json            # Re-ID performance metrics
├── baseline_membership_attack.json       # Membership attack results
├── baseline_attribute_attack.json        # Attribute attack results
├── privacy_utility_tradeoff.png          # Visualization of trade-offs
└── privacy_utility_summary.csv           # Summary table
```

### Expected Performance

**Baseline Model (No Privacy)**:
- mAP: ~75-80%
- Rank-1 Accuracy: ~85-90%
- Membership Attack AUC: ~0.65-0.75
- Attribute Attack Accuracy: ~70-80%

**DP Models** (privacy budget ε):
- ε = 1.0: Strong privacy, ~30-40% utility loss
- ε = 5.0: Moderate privacy, ~10-20% utility loss
- ε = 10.0: Mild privacy, ~5-10% utility loss

## Configuration

All hyperparameters can be configured in [config.py](config.py):

- Model architecture settings
- Training hyperparameters (baseline and DP)
- Privacy budgets to evaluate
- Image preprocessing parameters
- Attack configurations

## Project Structure

```
.
├── config.py                      # Configuration and hyperparameters
├── model.py                       # Re-ID model architecture
├── dataset.py                     # Market-1501 dataset loader
├── train_baseline.py              # Train baseline model
├── train_dp.py                    # Train DP model
├── extract_embeddings.py          # Extract feature embeddings
├── evaluate_reid.py               # Evaluate Re-ID performance
├── attack_membership.py           # Membership inference attack
├── attack_attribute.py            # Attribute inference attack
├── privacy_utility_tradeoff.py    # Full analysis pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Privacy Guarantees

This implementation provides **(ε, δ)-differential privacy** guarantees:

- **ε (epsilon)**: Privacy budget - lower values = stronger privacy
- **δ (delta)**: Probability of privacy breach (set to 1e-5)

The privacy guarantees are tracked using **Opacus** library's privacy accounting, which implements:
- Per-sample gradient clipping
- Gaussian noise addition to gradients
- Privacy budget tracking via Rényi Differential Privacy

## Performance Tips

### For CPU Training
- Reduce batch size (e.g., `--batch-size 16`)
- Reduce number of epochs
- Use smaller epsilon values (faster convergence)

### For GPU Training
- Increase batch size for better GPU utilization
- Consider data parallel training for multiple GPUs

### Memory Optimization
- Reduce image size in `config.py`
- Lower batch size
- Use gradient accumulation

## Troubleshooting

### Dataset Issues
- **Error**: "Dataset directory not found"
  - **Solution**: Ensure you've downloaded and extracted Market-1501 dataset to `data/Market-1501/`

### Memory Issues
- **Error**: "CUDA out of memory" or system hanging
  - **Solution**: Reduce batch size in config.py or use `--batch-size` flag

### Training Issues
- **Error**: DP training very slow
  - **Solution**: This is normal - DP-SGD is computationally expensive. Consider using GPU or reducing dataset size for testing

### Import Errors
- **Error**: "No module named 'opacus'"
  - **Solution**: Install missing dependencies: `pip install -r requirements.txt`

## Citation

If you use this code for research, please cite:

```bibtex
@misc{privacy-aware-reid,
  author = {Dhanjay Garg},
  title = {Privacy-Aware Person Re-Identification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dhanjo/Privacy-Aware}
}
```

### Dataset Citation

```bibtex
@inproceedings{zheng2015scalable,
  title={Scalable Person Re-identification: A Benchmark},
  author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2015}
}
```

## References

- [Market-1501 Dataset](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- [Differential Privacy in Deep Learning (Opacus)](https://github.com/pytorch/opacus)
- [Person Re-Identification: Past, Present and Future](https://arxiv.org/abs/1610.02984)
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [dhanjo].

## Acknowledgments

- Market-1501 dataset authors
- PyTorch and Opacus teams
- Open-source computer vision community
