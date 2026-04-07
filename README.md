# 🎬 Movie Recommendation System (Linear Algebra Based)

This project implements a **Movie Recommendation System** using core concepts from **Linear Algebra**.
It predicts user preferences and suggests movies by combining **Collaborative Filtering** and **Content-Based Filtering** techniques.

---

## 🚀 Features

* 📊 User–Movie Rating Matrix construction
* ⚙️ Gaussian Elimination & LU Decomposition
* 🧠 Gram-Schmidt Orthogonalization (independent taste factors)
* 📉 Singular Value Decomposition (SVD)
* 📈 Least Squares Prediction
* 🔀 Hybrid Recommendation System (CF + Content-Based)
* 💬 Interactive CLI for real-time recommendations

---

## 🧮 Concepts Used

* Matrices and Linear Transformations
* Gaussian Elimination
* LU Decomposition
* Vector Spaces (Row Space, Column Space, Null Space)
* Linear Independence & Basis
* Gram-Schmidt Orthogonalization
* Singular Value Decomposition (SVD)
* Least Squares Method

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn
```

### 2. Run the program

```bash
python movie_recommender.py
```

---

## 📂 Dataset

This project uses the **MovieLens 100K dataset**.

🔗 Download here:
https://grouplens.org/datasets/movielens/100k/

### After downloading:

Extract the dataset and place it like this:

```
project_folder/
│
├── movie_recommender.py
└── ml-100k/
    ├── u.data
    ├── u.item
```

### ⚠️ Note

* If the dataset is **not found**, the program will automatically generate a **synthetic dataset**.
* Only `u.data` and `u.item` are required.

---

## 💡 How It Works

1. Builds a **User–Movie Rating Matrix**
2. Applies **Linear Algebra techniques** to analyze user preferences
3. Uses:

   * **SVD** → find hidden patterns
   * **Least Squares** → predict missing ratings
   * **Content similarity (TF-IDF)** → match genres
4. Combines results into a **hybrid recommendation score**
5. Outputs **Top-N recommended movies**

---

## 📌 Example Output

```
TOP 10 RECOMMENDATIONS
1. Movie Name [Action, Sci-Fi] ███████ 0.89
2. Movie Name [Drama] ██████ 0.85
```

---

## 🎯 Applications

* OTT platforms (Netflix, Amazon Prime)
* E-commerce recommendations
* Personalized content systems

---

## 📚 Academic Relevance

This project demonstrates practical applications of:

* Linear Algebra in Machine Learning
* Recommendation Systems
* Data Science workflows

---

## 👨‍💻 Author

**Abhiram**
BTech CSE Student

---

## ⭐ Acknowledgment

Dataset provided by **MovieLens (GroupLens Research)**.
