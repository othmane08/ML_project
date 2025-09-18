# 🚀 Churn Prediction API  

A Machine Learning project deployed with **FastAPI** that predicts customer churn using trained models.  
The API provides endpoints to make predictions for a single customer or in bulk.  

---

## 📌 Features  
- 🔮 Predict whether a customer will churn or not.  
- 📦 Bulk prediction for multiple customers.  
- 📊 Pretrained ML pipeline (scikit-learn).  
- 🌐 Interactive API documentation with **Swagger UI** (`/docs`).  

---

## 🛠️ Tech Stack  
- **Python 3.10**  
- **FastAPI**  
- **Uvicorn**  
- **Scikit-learn**  
- **Joblib**  

---

## ⚙️ Installation & Usage  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/othmane08/ML_project.git
cd ML_project

### 2️⃣ Create & Activate Virtual Environement
conda create -n myenv python=3.10 -y
conda activate myenv

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run the API
uvicorn app.main:app --reload

### 5️⃣ Access the API
Swagger UI: http://127.0.0.1:8000/docs
ReDoc:http://127.0.0.1:8000/redoc

📂 Project Structure
ML_project/
│── app/
│   ├── main.py               # FastAPI application
│   ├── __init__.py
│── churn_pipeline.pkl        # Trained ML model
│── feature_columns.pkl       # Feature metadata
│── requirements.txt          # Dependencies
│── notebooks/                # Jupyter notebooks (exploration & training)

📌 API Endpoints
🔹 Root
GET / → Welcome message.

🔹 Predict (Single)

POST /predict

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MonthlyCharges": 70.35,
  "TotalCharges": 350.5
}

🔹 Predict Bulk

POST /predict_bulk

[
  {
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MonthlyCharges": 80.5,
    "TotalCharges": 160.5
  },
  {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 20,
    "PhoneService": "No",
    "MonthlyCharges": 50.0,
    "TotalCharges": 1000.0
  }
]

📊 Example Prediction Response
{
  "churn": "Yes",
  "probability": 0.87
}
📜 License

This project is licensed under the MIT License
👨‍💻 Author

Developed by Othmane Idrissi Sbai
