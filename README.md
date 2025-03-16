ITCZ Deep Learning Research

🌍 Project Overview

This repository contains research on the Inter-Tropical Convergence Zone (ITCZ) using Deep Learning techniques. The ITCZ is a significant meteorological phenomenon that influences global climate patterns, rainfall distribution, and extreme weather events. By applying Convolutional Neural Networks (CNNs) and Machine Learning algorithms, this project aims to analyze ITCZ behavior and improve climate forecasting models.


☀️ Understanding ITCZ and Climate Change

The Inter-Tropical Convergence Zone (ITCZ) is a key factor in global climate systems, directly affecting rainfall, droughts, and storm formation. Changes in ITCZ behavior can lead to shifts in monsoon patterns, which impact agriculture, water supply, and disaster management in tropical and subtropical regions. By using deep learning models, we can analyze past and present ITCZ data to better understand its trends and predict future changes.

🌍📈 Bridging AI and Meteorology

Traditional weather forecasting relies on numerical models and historical data analysis. While these methods have been effective, they often struggle with complex climate variability and require intensive computing resources. AI-powered approaches, such as CNN-based pattern recognition, can identify hidden correlations in climate data that traditional methods might miss. This project demonstrates how machine learning can be leveraged to improve forecasting accuracy and complement traditional meteorological models.

🎯Real-World Problems This Project Helps Solve

✅ More Accurate Weather Forecasts – By training models on ITCZ shifts, meteorologists can enhance the accuracy of rainfall and storm predictions, leading to better disaster preparedness.

✅ Improved Agricultural Planning – Farmers and policymakers depend on seasonal rainfall predictions for crop planning. With more reliable ITCZ-based forecasting, they can mitigate risks associated with droughts or unexpected rainfall changes.

✅ Disaster Management & Climate Adaptation – Governments and climate organizations can use this research to prepare for extreme weather events such as hurricanes, typhoons, and prolonged droughts, reducing the economic and human impact of these disasters.

✅ Advancing Climate Research – This project provides a framework for AI-driven climate modeling, helping researchers further explore the impact of global warming on atmospheric circulation patterns.

📌 Key Features – Detailed Explanation
This project is built with a structured deep learning approach to analyze weather data, focusing on ITCZ (Inter-Tropical Convergence Zone) research. Below is a detailed breakdown of the key features and how each one contributes to the overall goal.


1️⃣ Deep Learning for Climate Prediction

🔹 How it Works:
The project uses Convolutional Neural Networks (CNNs) to analyze weather-related image data (e.g., satellite images, climate maps).
CNNs are powerful for pattern recognition and spatial data processing, making them ideal for weather forecasting.
The binary classification model is trained to recognize weather patterns influenced by ITCZ shifts.


🔹 Why It Matters:
Traditional weather models rely on statistical methods, but deep learning finds hidden patterns in data that humans might miss.
CNN-based models can predict changes in atmospheric circulation, helping meteorologists make more accurate predictions.


🔹 Technical Details:
The model architecture likely includes convolutional layers for feature extraction and fully connected layers for classification.
The binary cross-entropy loss function is used because the task involves classifying weather patterns into two categories.


2️⃣ Database-Driven Analysis

🔹 How it Works:
Instead of storing results in CSV files, this project uses an SQLite database (ITCZ_512_RELU.db) to log experimental results.
Each training experiment (optimizer, accuracy, loss, validation loss) is stored in a structured format for easy comparison.
The database allows researchers to query results and compare different training configurations.


🔹 Why It Matters:
Instead of manually tracking results, this approach allows automated performance logging.
Researchers can run SQL queries to find the best-performing model configuration.
Makes the workflow scalable—new training results can be stored and retrieved easily.


🔹 Technical Details:
The table ITCZ_512_RELU stores the following data:
Optimizer → Which optimizer was used
Learning_Rate → The step size for model updates
Loss_Function → (Binary Cross-Entropy)
Size_of_Model → The number of parameters
Accuracy, Loss, Validation_Loss → Performance metrics
Example SQL query to retrieve the top 3 best optimizers based on accuracy:
SELECT Optimizer, Accuracy FROM ITCZ_512_RELU ORDER BY Accuracy DESC LIMIT 3;


3️⃣ Multiple Optimizers Tested

🔹 How it Works:
The model is trained using different optimizers to see which one gives the best performance.
Each optimizer adjusts weights differently, affecting how fast and accurately the model learns.
After training, results are compared using the SQLite database.


🔹 Why It Matters:
Optimizers are critical in deep learning—picking the right one can significantly improve model accuracy.
Testing multiple optimizers ensures the best one is chosen for the specific dataset.
Adadelta performed best (~61.7% accuracy) in this project.


🔹 Technical Details:
Optimizers tested: Adam, RMSprop, Nadam, Adagrad, Adadelta, Adamax, Ftrl.
The best optimizer (Adadelta) showed the highest accuracy and the lowest loss, meaning it helped the model learn better.
Instead of manually comparing optimizers, the results are logged into the database for quick retrieval.


4️⃣ Image Processing & Augmentation

🔹 How it Works:
Image Augmentation is applied to increase dataset diversity and reduce overfitting.
Transformations include:
Rescaling: Normalizing pixel values to [0,1].
Horizontal Flipping: Helps with symmetry-based weather pattern detection.
Shifting (width & height): Helps the model recognize shifted weather patterns.
Zooming & Rotation: Prevents the model from over-relying on a fixed orientation of features.


🔹 Why It Matters:
Climate data is limited, and augmentation artificially increases the dataset size.
Prevents overfitting, allowing the model to generalize better on new weather images.
Makes the model robust to real-world variations in climate data.


🔹 Technical Details:
Uses Keras ImageDataGenerator() for augmentation:
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5
)
These transformations ensure that even if ITCZ patterns appear in slightly different positions, the model can still detect them.


5️⃣ Automated Training Pipeline

🔹 How it Works:
Instead of manually loading data and training the model step by step, this project automates the entire workflow.
The training script follows these steps:
Load dataset from Google Drive (for privacy reasons, dataset isn't public).
Preprocess images (resizing, normalization, augmentation).
Train CNN model with different optimizers.
Log results into SQLite database.
Evaluate performance (accuracy, loss curves).


🔹 Why It Matters:
Makes it easy to run multiple training experiments without manual intervention.
Instead of adjusting settings manually, the system automatically logs and compares different configurations.
Saves time and computational resources, allowing researchers to focus on interpreting results instead of managing code execution.


🔹 Technical Details:
The Jupyter Notebook (DBConnectionEstablished.ipynb) executes the entire workflow in one run.
The database acts as a tracking system, ensuring reproducibility of training results.
After training, accuracy/loss plots are automatically generated for analysis.

📂 Repository Structure

ITCZ_DeepLearning_Research/

│── Databases/     # Stores SQLite database with model results 

│── Machine_Learning_Algorithm  /# Contains model training and evaluation scripts

│── DBConnectionEstablished.ipynb # Jupyter Notebook with model training workflow

│── README.md    # Documentation (this file)

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

📊 Results & Insights – Optimizers in This Project
In this project, multiple optimizers were tested to determine which one yields the highest accuracy and lowest loss when training a deep learning model on ITCZ-related weather data. Optimizers are crucial in deep learning because they control how the model updates its weights during training, affecting both the speed and quality of learning.

🔹 What Are Optimizers?
Optimizers are algorithms that adjust the neural network’s weights based on the computed loss during training. The goal is to minimize the loss function, ensuring that the model learns efficiently and generalizes well to unseen data. In this project, Binary Cross-Entropy was used as the loss function, meaning the model is performing binary classification (e.g., predicting presence/absence of a pattern in weather data).

🔹 Optimizers Tested in This Project
This project evaluates several well-known optimizers, each with different characteristics and advantages:

1️⃣ Adam (Adaptive Moment Estimation)

How it works: Adam combines momentum-based and adaptive learning rate techniques. It keeps track of both the first moment (mean of gradients) and second moment (uncentered variance), adjusting learning rates for each parameter.
Strengths: Works well with noisy data, requires little fine-tuning.
Weaknesses: Sometimes overfits due to aggressive learning rate adaptation.
Results in this project: Performed consistently but had moderate accuracy (~49.8%).


2️⃣ RMSprop (Root Mean Square Propagation)

How it works: Divides the learning rate by a moving average of squared gradients, stabilizing updates.
Strengths: Works well in recurrent neural networks (RNNs) and in cases where data distributions change.
Weaknesses: Requires careful tuning of hyperparameters.
Results in this project: Lower accuracy (~49.2%) compared to Adam, but better stability.


3️⃣ Nadam (Nesterov-accelerated Adaptive Moment Estimation)

How it works: An improved version of Adam that incorporates Nesterov momentum, making weight updates slightly more predictive.
Strengths: Can converge faster than Adam in some cases.
Weaknesses: Similar overfitting risks as Adam.
Results in this project: Moderate accuracy (~52.0%), slightly better than Adam.


4️⃣ Adagrad (Adaptive Gradient Algorithm)

How it works: Assigns smaller learning rates to frequently updated parameters and higher learning rates to rarely updated ones.
Strengths: Good for sparse data.
Weaknesses: Learning rate can decay too much, leading to premature convergence.
Results in this project: Performed well (~60.1%), second highest accuracy among optimizers.


5️⃣ Adadelta (Improved Adagrad)

How it works: Similar to Adagrad but fixes the issue of decaying learning rates by computing learning rates dynamically.
Strengths: More adaptive than Adagrad, doesn’t require manual learning rate tuning.
Weaknesses: Computationally expensive.
Results in this project: Best optimizer (~61.7% accuracy), showing strong generalization.


6️⃣ Adamax (Variant of Adam)

How it works: A more stable version of Adam that uses infinity norm instead of L2 norm, making it work better for large-scale models.
Strengths: Handles high-dimensional data better.
Weaknesses: Not always better than Adam.
Results in this project: Lower accuracy (~49.8%), similar to Adam.


7️⃣ Ftrl (Follow The Regularized Leader)

How it works: Designed for large-scale problems with high-dimensional sparse data, commonly used in Google’s Ad click prediction models.
Strengths: Good for highly sparse data.
Weaknesses: Not always effective in deep learning problems.
Results in this project: Low accuracy (~49.8%), not ideal for this dataset.


🔹 Key Takeaways from Optimizer Testing


Adadelta was the best-performing optimizer, achieving ~61.7% accuracy, making it a strong choice for ITCZ prediction tasks.
Adagrad also performed well (~60.1%), benefiting from adaptive learning rates.
Adam, RMSprop, and Nadam had moderate results but were not as effective as Adadelta.
Ftrl and Adamax underperformed, likely due to the nature of the dataset.

📧 Contact

For questions or collaborations, reach out via GitHub Issues or email 16omerkilic@gmail.com
