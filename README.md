# Adaptive-RecSys-using-Continual-Learning

## Overview
This project focuses on building a recommendation system that can predict the next product a user will interact with during their session. The primary challenge is adapting to new trends while retaining older consumer behavior. The project also tackles problems like concept drift and catastrophic forgetting in recommendation systems, aiming for improved prediction accuracy over time.

### Key Features:
- Predict the next product a user will interact with.
- Adapt to evolving trends while retaining older consumer preferences.
- Mitigate concept drift and performance degradation.
  
---

## Problem Statement
The project addresses the following challenges:
- **Concept Drift**: Changes in user preferences over time.
- **Catastrophic Forgetting**: Loss of prior knowledge while adapting to new data.
- **Large Action Space & Dataset Size**: Scalability of the model to handle large datasets and actions.
- **Computational Efficiency**: Training and inference time due to large datasets.

---

## Dataset
The dataset used in this project is the **Ecommerce Behavior Data from Multi-Category Store**, available on Kaggle.

- **Dataset**: [Ecommerce Behavior Data from Multi-Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store/data)
- The dataset contains interactions between users and products across various categories. It includes features like user ID, product ID, sessions, and timestamps of interactions.

---

## Approach

### 1. **Sliding Window with Replay Buffer**
- **Objective**: Give more importance to the most recent user-product interactions while retaining a portion of historical data to prevent catastrophic forgetting.
- **Method**:
  - Use a sliding window to focus on recent data.
  - Implement a replay buffer to store a small percentage of past interactions, allowing the model to remember older data.
  
<img width="854" alt="image" src="https://github.com/user-attachments/assets/7af2c8ee-c665-4fab-bdea-24bf05b774c4">


#### Results:
Below are the Top-K accuracy results for the **Baseline LSTM** and **Sliding & Replay Buffer** approaches:

| **Data Size**         | **Testing Phases**         | **Baseline LSTM (Top K Accuracy)** | **Sliding & Replay Buffer (Top K Accuracy)** |
|-----------------------|----------------------------|------------------------------------|---------------------------------------------|
| **Smaller (10%)**     | Initial Set                | 11.72%                             | 11.91%                                      |
|                       | After New Data Ingestion   | 8.11%                              | 12.43%                                      |
| **Huge (1M interactions)** | Initial Set            | 21.56%                             | 21.15%                                      |
|                       | After New Data Ingestion   | 16.79%                             | 22.52%                                      |

### 2. **Transfer Learning**
- **Objective**: Use transfer learning to retain prior knowledge (by freezing lower layers) while fine-tuning upper layers with new data to adapt to evolving patterns.
- **Method**:
  - Freeze the lower layers to preserve historical knowledge.
  - Fine-tune the upper layers using the most recent data.

<img width="541" alt="image" src="https://github.com/user-attachments/assets/02459901-c901-4873-a644-a16a76b6d6b4">


#### Results:
Here are the Top-K accuracy results for the **Baseline LSTM** and **Transfer Learning** approaches:

| **Data Size**         | **Testing Phases**         | **Baseline LSTM (Top K Accuracy)** | **Transfer Learning (Top K Accuracy)** |
|-----------------------|----------------------------|------------------------------------|---------------------------------------|
| **Smaller (10%)**     | Initial Set                | 11.58%                             | 11.58%                                |
|                       | After shift of 15 days     | 7.45%                              | 12.81%                                |
| **Huge (1M interactions)** | Initial Set            | 23.56%                             | 23.56%                                |
|                       | After shift of 15 days     | 17.09%                             | 24.86%                                |

---

## Evaluation Metrics
- **Top-K Accuracy**: Measures how often the correct product is found within the top K predictions. This metric is crucial for recommendation systems.

---
## Results & Analysis

### Key Findings:
- The **Sliding Window with Replay Buffer** method outperforms the baseline LSTM in adapting to new data, with consistent performance even after new data ingestion.
- **Transfer Learning** shows better stability over time compared to the baseline models, especially in the face of concept drift.

### Challenges:
- Handling **large datasets** and maintaining **computational efficiency** were significant challenges.
- The **complexity of managing large action spaces** for recommendation systems. The insutry standard for top-k approaches with large action spaces is 20-30%.

### Future Work:
- Experiment with other **continual learning techniques** to further improve performance.
- Implement additional **optimization strategies** for computational efficiency.
