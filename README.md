# CVTC-M: Multimodal Model for Alzheimer’s Early Diagnosis

#### Project Overview
This project develops a lightweight multimodal fusion model (CVTC-M) for the early diagnosis and conversion time prediction of Alzheimer’s disease (AD) by integrating MRI, protein concentrations, APOE genotypes, and SNP data. Leveraging an implicit time-series modeling approach with scale-adaptive embedding and long-short attention mechanisms, the model enhances prediction accuracy while reducing computational complexity.

#### Features
- Multimodal Data Integration: Combines MRI images, protein concentrations, APOE, and SNP data for AD early diagnosis.
- Implicit Time-Series Modeling: Captures disease dynamics using final time-point labels without complex time-series alignment.
- High-Precision Prediction: Achieves 86.32% accuracy and 1.09-year MAE on ADNI, and 85.34% accuracy and 1.28-year MAE on NACC.
- Biological Mechanism Insights: Analyzes SNP-APOE and SNP-protein interactions, constructs an APOE-CLU-HSP60-BRCA1 network, and identifies potential biomarkers.
- Lightweight Design: Features ~6.51M parameters and 0.0207-second inference time, suitable for resource-constrained devices.

#### Installation
1. Ensure Python 3.12.4 or higher is installed.
2. Clone the repository:
   ```bash
   git clone https://github.com/username/CVTC-M.git
   ```
3. Install dependencies:
   ```bash
   pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.0 torch==2.5.0 scipy==1.14.0 nibabel==5.3.2 matplotlib==3.8.4
   ```
4. Install GPU-supported PyTorch (optional):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

#### Usage
1. Data Preparation: Place ADNI and NACC datasets in the `Associated` or `CVTC` folders, ensuring format compliance with the paper (MRI, SNP, APOE, protein data).
2. Run the Model:
   - Navigate to the CVTC directory and run the main script:
     ```bash
     python main.py --data_path ./CVTC --output_dir ./results
     ```
   - Arguments: `--data_path` is the input data path, `--output_dir` is the output results path.
3. Result Analysis: Outputs include conversion type predictions (accuracy, AUC) and conversion time predictions (MAE, etc.), detailed in the `results` folder.

#### Data and Dependencies
- Datasets: ADNI (377 patients for training, MRI, APOE, SNP, protein data) and NACC (100 patients for testing) datasets, available at [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/) and [https://naccdata.org/](https://naccdata.org/). Single-cell RNA-seq data from GEO (GSE264648).
- Dependencies: Python 3.12.4, pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.0, PyTorch 2.5.0, etc., see installation guide for details.

#### Contributing
Contributions are welcome via Issues or Pull Requests! Please refer to the [Contributing Guidelines](CONTRIBUTING.md) for details.

#### License
This project is released under the [MIT License](LICENSE).

#### References
See the "References" section of the paper "CVTC-M: Multimodal Model for Alzheimer’s Early Diagnosis with Interaction Mechanism Insights" for details.
