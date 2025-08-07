# Sentiment Analysis on Amazon Alexa Reviews
This project focuses on analyzing customer reviews for Amazon Alexa products to determine whether the sentiment expressed is **positive or negative**. It uses NLP techniques and machine learning models to build a reliable sentiment classification pipeline.

## 🔍 Project Objectives

- Understand customer sentiment from review text.
- Preprocess and clean textual data using NLP.
- Train and evaluate classification models.
- Visualize key insights using WordClouds and plots.

## 📂 Dataset

The dataset used contains reviews for Amazon Alexa products with associated ratings and labels for sentiment. It includes:
- Review text
- Feedback (1 = Positive, 0 = Negative)
- Ratings

## 🛠️ Tools & Libraries

- **Python**  
- **Pandas, NumPy, Matplotlib, Seaborn**  
- **NLTK** for text preprocessing  
- **Scikit-learn** (RandomForest, DecisionTree)  
- **XGBoost** for advanced classification  
- **WordCloud** for visualization  

## 🔄 Workflow

1. **Text Cleaning**: Lowercasing, removing punctuation, stopwords, and stemming.
2. **Feature Engineering**: Using `CountVectorizer` for bag-of-words model.
3. **Model Training**: Tried multiple classifiers including:
   - Decision Tree
   - Random Forest
   - XGBoost
4. **Evaluation**: Confusion matrix, accuracy, and cross-validation scores.
5. **Model Saving**: Trained models are saved using `pickle`.

## 📊 Visualizations

- WordClouds of frequent words in positive and negative reviews.
- Rating distributions and feedback insights.
- Confusion matrices for model performance.

## ✅ Results

- Built multiple classification models with strong accuracy.
- Identified common keywords and patterns in customer feedback.
- Showed the effectiveness of preprocessing in NLP tasks.

## 🚀 How to Run

1. Clone the repository.
2. Open the `Sentiment_analysis.ipynb` notebook.
3. Ensure the dataset is available or load your own.
4. Run the notebook cells in order.

## 📌 Future Improvements

- Incorporate deep learning models like LSTM.
- Use TF-IDF or word embeddings instead of CountVectorizer.
- Build a web app to deploy the model for real-time predictions.

---

## 📄 License

Open-source under [MIT License](LICENSE).
