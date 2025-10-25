# Tugas Individu 1 IF5152

This project follows the spesification given for the features using **OpenCV**, **Scikit-Image**, and **NumPy**: 
- Gaussian Filtering  
- Sobel Edge Detection  
- Canny Edge Detection  
- Harris Corner Detection  
- SIFT dan FAST Feature Detection  
- Camera calibration

It's recommended to run all the script using **Python virtual environment (venv)**.

---

## 📦 Requirements

- Python ≥ 3.8  
- pip 
- git (optional)

---

## 🚀 Installation & Preparation

1. **Clone this repository**
   ```bash
   git clone https://github.com/Flame25/Christopher-Richard_18222057_IF5152_TugasIndividuCV.git
   cd Christopher-Richard_18222057_IF5152_TugasIndividuCV
   ```
2. **Prepare venv**
   ```bash 
   python3 -m venv venv
   source venv/bin/activate        # Linux / macOS
   # venv\Scripts\activate         # Windows (PowerShell)
   ```
3. **Install Dependancies**
    ```bash 
    pip install -r requirements.txt
    ```

## How to Run 

1. **Go to selected folders** 
2. **Run the python scripts** 
    ```bash
    python script_name.py
    ```
All processed images and results will be saved in the directory, organized by their respective processing stage (e.g., gaussian_filter, sobel_filter, etc.).


## Project Structure 
```
Nama_NIM_IF5152_TugasIndividuCV//
│
├── 01_filtering/
│   ├── custom_dataset/
│   ├── numpy_results/
│   ├── scikit_results/
│   └── script.py
├── 02_edge/
├── 03_featurepoints/
├── 04_geometry/
├── 05_laporan.pdf
├── venv/                     # Virtual environment (excluded in .gitignore)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
