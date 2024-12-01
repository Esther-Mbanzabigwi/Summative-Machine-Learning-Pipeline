# Plant Disease Recognition System
--------------------------------------------------------------------

## **Project Description**

The Plant Disease Recognition System is an advanced machine learning platform designed to assist farmers, agricultural experts, and researchers in identifying diseases in plants with precision and ease. Leveraging state-of-the-art deep learning models, this platform processes plant images and provides accurate disease classifications, empowering users to take timely and informed actions to protect crop health and maximize agricultural yield.

This project implements solution to detect plant diseases from images. It includes:
- Preprocessing steps for image data.
- Model training and testing.
- A deployment package (public URL with Docker, mobile app, or desktop app).
- Flood request simulation to test system performance.

## **Dataset**

[Plant diseases dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
---

## **Demo**
- **Video Demo**: [YouTube Demo Link](https://www.loom.com/share/eb05a486cb61422d922b6fad55078808?sid=3e74c0d8-6add-4e84-aa5e-4c3024a95f49)  
- **Deployed URL (if applicable)**: [Plant Disease Recognition System](https://summative-mlop.onrender.com/)

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/Esther-Mbanzabigwi/Summative-Machine-Learning-Pipeline
cd Summative-Machine-Learning-Pipeline
```

### **2. Install Dependencies**
Install Python dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### **3. Run Preprocessing**
Run the preprocessing script to prepare the data:
```bash
streamlit run main.py --server.port=8501 --server.enableCORS=false
```

### **4. Train the Model**
Train the model using:
```bash
python src/training.py
```

### **5. Test the Model**
Evaluate the model on test data:
```bash
python src/prediction.py
```

## **Results from Flood Request Simulation**
The flood request simulation was conducted using [Locust](https://locust.io). The results are summarized below:

- **Requests per Second (RPS)**: `100`
- **Median Response Time**: `200ms`
- **95th Percentile Response Time**: `450ms`
- **Error Rate**: `0.5%`

**Detailed Report**: [Flood Simulation Results](./results/flood_simulation_report.html)

---

## **Notebook**
The Jupyter Notebook contains:
1. Preprocessing functions.
2. Model training and testing.
3. Prediction functions.

**File**: [Plant_Disease_Recognition_Notebook.ipynb](https://github.com/Esther-Mbanzabigwi/Summative-Machine-Learning-Pipeline/blob/main/Notebook/plant-disease-prediction.ipynb)

---

## **Model Files**
- **Pickle File (.pkl)**: [Model File](./model/best_checkpoint.pkl)
- **TensorFlow File (.ptf)**: [Model File](./model/best_checkpoint.pth)

---

## **Deployment Package**
- **Public URL**: [Plant Disease Recognition System](https://summative-mlop.onrender.com/)
- **Docker Image**: [Docker Hub Link](https://github.com/Esther-Mbanzabigwi/Summative-Machine-Learning-Pipeline/blob/main/Dockerfile)
