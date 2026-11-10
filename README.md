# Movie Recommendation System

Predicting how users will rate movies on the **MovieLens 100k** dataset, comparing a regression baseline with user-based and item-based collaborative filtering and simple ensembles.

📓 **Notebook:** [`movie_recommendation.ipynb`](movie_recommendation.ipynb) · 📄 **Report:** [`reports/movie_recommendation_report.pdf`](reports/movie_recommendation_report.pdf)

## Approach

1. **Exploration** – 100,000 ratings from ~940 users on ~1,660 movies; user demographics (age bands, occupation), most-rated movies and the rating distribution.
2. **Preprocessing** – encoding categorical features, z-score scaling, train/test split.
3. **Regression baseline** – linear regression, then Lasso (L1) and Ridge (L2) tuned with `GridSearchCV`.
4. **Collaborative filtering** – user-based and item-based KNN with [Surprise](https://surpriselib.com/) using Pearson-baseline similarity; `k` chosen with the elbow method.
5. **Ensembles** – averaging the three models, and a linear regression that uses the CF predictions as features.

## Results (test set)

| Model | MAE |
| :--- | :--- |
| Linear regression | 0.777 (RMSE 0.977) |
| User-based CF (k = 10) | 0.802 |
| Item-based CF (k = 10) | **0.784** |
| Averaging ensemble | 0.872 |

Regularisation did not beat plain linear regression, suggesting every feature carries useful signal. Item-based CF beat user-based CF at every `k`. The full discussion is in the report.

## Running

```bash
pip install -r requirements.txt
jupyter notebook movie_recommendation.ipynb
```

The dataset is included at `data/movies_ratings_100k.csv` (MovieLens 100k, [GroupLens](https://grouplens.org/datasets/movielens/100k/)).
