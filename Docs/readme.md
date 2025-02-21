# Bot Detection System for Social Media

## Overview
This project is a machine learning-based bot detection system for social media. It leverages advanced algorithms, including hierarchical models, to differentiate between human and bot accounts based on behavioral patterns, text analysis, and engagement metrics.

## Features

- **Hierarchical Model**: Implements a multi-stage classification system using behavioral analysis, text-based analysis, and a meta-classifier for enhanced accuracy.
- **Automated Detection**: Uses machine learning models (Random Forest, NLP, and Meta-classification) to detect bots on social media platforms.
- **Scalability**: Can be extended to various social media platforms.
- **Customizable**: Allows fine-tuning of detection thresholds and model parameters.
- **Adaptability**: Can handle dynamic bot behaviors using advanced anomaly detection techniques.
- **Visualization**: Generates feature importance rankings and confusion matrices for model evaluation.
- **Interactive Interface**: Integrates with Gradio for real-time bot prediction.

## Prerequisites
Before running the project, ensure you have the following installed:

- **Python**: Version 3.8+
- **Libraries**:
  ```
  numpy
  pandas
  scikit-learn
  seaborn
  matplotlib
  gradio
  requests
  beautifulsoup4
  torch
  transformers
  ```

## Usage

1. **Preprocess Data**: Prepare your dataset using the `preprocessing.py` script.
2. **Train the Model**: Use the `HierarchicalBotDetector` class to train the behavioral model, tweet content model, and meta-classifier.
3. **Test the Model**: Evaluate the trained models using test data and analyze performance metrics (accuracy, confusion matrix, feature importance).
4. **Deploy the Model**: Use the provided Gradio interface for real-time bot detection.

## Model Explanation

- Uses a **hierarchical approach** combining:
  - **Behavioral Analysis**: Features like account age, engagement metrics, and profile details.
  - **Text-Based Analysis**: NLP techniques (TF-IDF and transformers) to analyze tweet content.
  - **Meta-Classifier**: Integrates outputs from base models for improved accuracy.
- **Feature Engineering**:
  - Extracts features like activity patterns, text analysis, and network behavior.
  - Implements anomaly detection techniques to identify suspicious engagement patterns.
- **Training Approach**:
  - Trained on a labeled dataset of bots and human accounts.
  - Uses supervised learning methods (Random Forest, SVM, and Neural Networks).
  - Supports real-time adaptation to new bot behaviors.
- **Visualization**:
  - Generates confusion matrices and feature importance plots for better interpretability.

## Interactive Interface
The project includes a **Gradio-based web interface** for real-time bot detection. Users can input profile details and tweets to obtain a classification result with confidence scores.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## Contact
For any queries, reach out at singhash058@gmail.com.
