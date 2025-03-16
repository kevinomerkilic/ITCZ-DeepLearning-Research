ITCZ Deep Learning Research

🌍 Project Overview

This repository contains research on the Inter-Tropical Convergence Zone (ITCZ) using Deep Learning techniques. The ITCZ is a significant meteorological phenomenon that influences global climate patterns, rainfall distribution, and extreme weather events. By applying Convolutional Neural Networks (CNNs) and Machine Learning algorithms, this project aims to analyze ITCZ behavior and improve climate forecasting models.

📌 Key Features

Deep Learning for Climate Prediction: Utilizes CNN models to analyze weather data.

Database-Driven Analysis: Stores experimental results in an SQLite database for comparison.

Multiple Optimizers Tested: Evaluates Adam, RMSprop, Nadam, and more to find the best optimizer.

Image Processing & Augmentation: Applies transformations for improved model generalization.

Automated Training Pipeline: Includes scripts for data loading, augmentation, model training, and evaluation.

📂 Repository Structure

ITCZ_DeepLearning_Research/
│── Databases/                 # Stores SQLite database with model results
│── Machine_Learning_Algorithm/ # Contains model training and evaluation scripts
│── DBConnectionEstablished.ipynb # Jupyter Notebook with model training workflow
│── README.md                   # Documentation (this file)

🚀 How to Run the Project

1️⃣ Clone the Repository

git clone https://github.com/kevinomerkilic/ITCZ-DeepLearning-Research.git
cd ITCZ-DeepLearning-Research

2️⃣ Install Dependencies

pip install tensorflow pandas matplotlib sqlite3

3️⃣ Run the Jupyter Notebook

jupyter notebook DBConnectionEstablished.ipynb

4️⃣ Query the Database (Optional)

To inspect model results stored in SQLite, run:

sqlite3 Databases/ITCZ_512_RELU.db
SELECT * FROM ITCZ_512_RELU LIMIT 10;

📊 Results & Insights

Best Performing Optimizer: The results in the database help determine which optimizer yields the highest accuracy and lowest loss.

Validation Performance: The notebook contains accuracy and loss curves to visualize training behavior.

Data Augmentation Effectiveness: Includes an analysis of augmentation techniques on model generalization.

💡 Future Enhancements

Expand dataset for increased generalization.

Implement LSTM models for time-series climate forecasting.

Deploy the model as an API for real-time weather prediction.

🤝 Contributing

Contributions are welcome! If you have ideas for improvements, feel free to submit a pull request.

📜 License

This project is licensed under the MIT License – feel free to use and modify the code as needed.

📧 Contact

For questions or collaborations, reach out via GitHub Issues or email kevinomerkilic@yourdomain.com.
