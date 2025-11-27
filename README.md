🧠 Intelligent Stress Assessment System
Keystroke Dynamics–Based Real-Time Stress Detection Using Machine Learning

This project presents a non-intrusive, real-time stress detection system using keystroke dynamics. Unlike traditional stress measurement techniques requiring ECG, GSR, HRV, or EEG sensors, this system analyzes typing behavior to predict stress levels with high accuracy—making it practical for large-scale deployment in workplaces, education, mental health platforms, and general human–computer interaction systems.

The system includes:

A Streamlit dashboard

A 96.4% accuracy SVM model

A complete ML pipeline (preprocessing → feature engineering → model training → prediction)

Interactive typing tests, visualizations, and recommendations

🚀 Features
🔍 Real-Time Stress Prediction

Uses six keystroke-based behavioral features:

Typing Speed (WPM)

Error Rate (%)

Backspace Count

Hold Time (ms)

Flight Time (ms)

Pause Count

📊 Streamlit Dashboard

Real-time typing test with timer

Manual stress input prediction

Model comparison

Feature explanations

Confidence-based results

Personalized recommendations

🤖 Machine Learning Backend

Three ML models were trained and compared:

SVM (RBF Kernel) — Best accuracy: 96.4%

Random Forest

Logistic Regression

🧾 Technologies Used

Python

scikit-learn

NumPy / Pandas

Streamlit

Plotly

📂 Project Structure
📁 Intelligent-Stress-Assessment-System
│── dashboard.py             # Streamlit UI
│── stress_detection_model.pkl
│── scaler.pkl
│── label_encoder.pkl
│── requirements.txt
│── README.md
│── dataset.csv (optional)
└── ...

⚙️ Installation Guide
1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Create virtual environment
python -m venv venv


Activate:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

3️⃣ Install required packages
pip install -r requirements.txt

4️⃣ Run the Streamlit application
streamlit run dashboard.py


Your browser will automatically open the dashboard.

🧪 Dataset Description

The dataset contains:

150 samples

6 keystroke features

3 stress classes: low, medium, high

Each sample represents a complete typing session, with features engineered from keystroke logs.

📈 Model Performance
Model	Accuracy	Precision	F1-Score
SVM (RBF Kernel)	96.4%	0.95	0.96
Random Forest	91.2%	0.90	0.91
Logistic Regression	88.3%	0.87	0.88

SVM performed best due to its ability to capture non-linear patterns in keystroke timing behavior.

🧠 Why Keystroke Dynamics for Stress Detection?

Stress affects:

motor coordination

cognitive load

reaction time

error frequency

These changes naturally appear in typing behavior, making keystrokes a powerful, zero-cost biomarker.

Advantages over physiological & EEG systems:
Method	Accuracy	Cost	Intrusiveness	Real-Time
ECG / GSR Sensors	92–97%	High	High	Moderate
EEG	95–98%	Very High	Very High	Low
Wearable HRV	85–93%	Medium	Medium	High
Keystrokes (This System)	96.4%	Zero	None	High
🌐 Deployment (Streamlit Cloud)

This project is optimized for Streamlit Cloud hosting.

Steps:

Push repo to GitHub

Go to: https://share.streamlit.io

Click "Deploy App"

Select your repo & choose dashboard.py

Deploy—your app gets a public URL.

📝 Research Paper (IEEE Format)

This system is part of a research study titled:

"Real-Time Stress Detection Using Keystroke Dynamics and Machine Learning"

It compares the proposed behavioral approach with established physiological and EEG methods.

🤝 Contributing

Contributions and suggestions are welcome!

Fork the repository

Create a new branch

Submit a pull request

📜 License

This project is licensed under the MIT License.

⭐ Support

If you like this project, please ⭐ star the repository!
