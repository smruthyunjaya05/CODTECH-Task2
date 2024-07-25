# CODTECH-Task2

## Project Information

- **Name**: Mruthyunjaya S
- **Company**: CODTECH IT Solutions
- **ID**: CT4ML4545
- **Domain**: Machine Learning
- **Duration**: July to August 2024

## Project Overwiew
**Project**: Analysis On Movie Reviews

This project demonstrates the implementation of sentiment analysis on movie reviews using the IMDB dataset. The aim is to classify reviews as either positive or negative based on the text content, leveraging natural language processing (NLP) techniques and machine learning models.

## Objective

The primary goal of this project is to develop and evaluate a sentiment classification model using various NLP and machine learning techniques. The model will categorize reviews from the IMDB dataset into positive or negative sentiments. Key tasks include data cleaning, text processing, feature extraction, and model evaluation.

## Data Preparation and Exploration

The IMDB dataset contains 50,000 movie reviews labeled as positive or negative. Initial exploration involved:

- Checking for missing values and data types.
- Visualizing sentiment distribution using a count plot.
- Displaying example reviews with their sentiments.
- Analyzing word count and review length distributions for each sentiment class.

## Data Preprocessing

The text data was processed using the following steps:

1. **Lowercasing**: Converted all text to lowercase to maintain uniformity.
2. **HTML Tags and URLs Removal**: Stripped out any HTML tags and URLs using regular expressions.
3. **Special Characters and Punctuation Removal**: Eliminated non-alphanumeric characters.
4. **Tokenization**: Split the text into individual words.
5. **Stopwords Removal**: Removed common English stopwords using NLTK's stopwords list.
6. **Stemming**: Reduced words to their root form using the Porter Stemmer.

## Exploratory Data Analysis

- **Word Cloud**: Generated word clouds to visualize the most frequent words in positive and negative reviews.
- **Common Words Analysis**: Identified and plotted the top 15 most common words in positive and negative reviews using bar charts.

## Feature Extraction

- **TF-IDF Vectorization**: Transformed the cleaned text data into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.

## Model Implementation

1. **Data Splitting**: The dataset was split into training and testing sets using a 70-30 ratio.

2. **Model Training and Evaluation**:
   - **Logistic Regression**: Trained on the TF-IDF features to classify sentiments.
   - **Multinomial Naive Bayes**: Used to exploit the multinomial distribution of the feature counts.
   - **Support Vector Machine (SVM)**: Implemented with a linear kernel for high-dimensional feature space.

3. **Model Evaluation**: Assessed models using accuracy, confusion matrix, and classification report metrics.

4. **Hyperparameter Tuning**: Optimized SVM model using GridSearchCV for better performance.

## Results

- **Logistic Regression**:
  - Achieved a test accuracy of approximately [Logistic Regression Accuracy]%.
  - Confusion matrix and classification report highlight the precision, recall, and F1-score.

- **Multinomial Naive Bayes**:
  - Achieved a test accuracy of approximately [Multinomial Naive Bayes Accuracy]%.
  - Provided confusion matrix and classification report.

- **Support Vector Machine (SVM)**:
  - Initial test accuracy of approximately [SVM Initial Accuracy]%.
  - Best cross-validation score of approximately [GridSearchCV Best Score]% with optimal parameters.
  - Final test accuracy of approximately [SVM Final Accuracy]% after hyperparameter tuning.

## Conclusion

The sentiment analysis project successfully demonstrated how to process and analyze text data for classification tasks. The models, especially the tuned SVM, showed promising accuracy in classifying sentiments of movie reviews. Future work could explore deep learning techniques or ensemble methods to enhance prediction performance further.

This project provides valuable insights into the practical application of NLP and machine learning in sentiment analysis, emphasizing data preprocessing, feature extraction, model selection, and evaluation.

## Getting Started

To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
