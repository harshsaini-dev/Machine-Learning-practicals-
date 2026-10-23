"""Lab 07: customer K-Means segmentation and TF-IDF news clustering."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

BASE = Path(__file__).resolve().parent
DATA = BASE / 'datasets'; DATA.mkdir(exist_ok=True)
OUT = BASE / 'outputs' / 'lab_07'; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

def customers():
    blocks = []
    for age, income, score, purchases in [(24, 32, 85, 15), (48, 95, 35, 55), (35, 58, 60, 35)]:
        blocks.append(pd.DataFrame(dict(age=np.clip(rng.normal(age, 7, 80), 18, 75),
            annual_income_thousands=np.clip(rng.normal(income, 12, 80), 10, 180),
            spending_score=np.clip(rng.normal(score, 12, 80), 1, 100),
            annual_purchases=np.clip(rng.normal(purchases, 7, 80), 1, 90))))
    frame = pd.concat(blocks, ignore_index=True)
    frame.to_csv(DATA / 'lab_07_customers.csv', index=False)
    scaler = StandardScaler(); X = scaler.fit_transform(frame)
    scores = []
    for k in range(2, 8):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        scores.append(dict(k=k, silhouette=silhouette_score(X, model.labels_), inertia=model.inertia_))
    metrics = pd.DataFrame(scores)
    metrics.to_csv(OUT / 'customer_k_selection.csv', index=False)
    k = int(metrics.loc[metrics.silhouette.idxmax(), 'k'])
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    hierarchical = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    print('Customer clusters:', k, 'KMeans silhouette:', round(silhouette_score(X, model.labels_), 3),
          'Hierarchical silhouette:', round(silhouette_score(X, hierarchical), 3))
    frame['cluster'] = model.labels_
    frame.to_csv(OUT / 'customer_segments.csv', index=False)
    frame.groupby('cluster').mean().round(1).to_csv(OUT / 'customer_profiles.csv')
    joblib.dump(dict(scaler=scaler, model=model), OUT / 'customer_kmeans.joblib')
    plt.figure(); plt.scatter(frame.annual_income_thousands, frame.spending_score, c=model.labels_, cmap='viridis')
    plt.xlabel('Income (thousands)'); plt.ylabel('Spending score'); plt.title('Customer clusters')
    plt.tight_layout(); plt.savefig(OUT / 'customer_clusters.png'); plt.close()
    plt.figure(); plt.plot(metrics.k, metrics.silhouette, marker='o'); plt.xlabel('k'); plt.ylabel('Silhouette')
    plt.tight_layout(); plt.savefig(OUT / 'customer_k_selection.png'); plt.close()

def news():
    topics = dict(technology=['software developers build artificial intelligence tools',
        'computer security researchers improve network protection',
        'technology companies release cloud computing platforms',
        'engineers design faster processors for mobile devices'],
        sports=['football team wins championship after final',
        'cricket players prepare for international tournament',
        'tennis athlete trains for major competition',
        'basketball coach discusses defensive team strategy'],
        finance=['stock market investors evaluate company earnings',
        'central bank discusses inflation and interest rates',
        'business analysts review quarterly revenue reports',
        'economic growth influences global currency markets'])
    frame = pd.DataFrame([dict(article=items[i % len(items)] + f' article edition {i%3}', reference_topic=topic)
                          for topic, items in topics.items() for i in range(12)])
    frame.to_csv(DATA / 'lab_07_news_articles.csv', index=False)
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = tfidf.fit_transform(frame.article)
    scores = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        scores.append(dict(k=k, cosine_silhouette=silhouette_score(X, labels, metric='cosine')))
    metrics = pd.DataFrame(scores); metrics.to_csv(OUT / 'news_k_selection.csv', index=False)
    k = int(metrics.loc[metrics.cosine_silhouette.idxmax(), 'k'])
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    ag = AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine').fit_predict(X.toarray())
    print('News clusters:', k, 'KMeans silhouette:', round(silhouette_score(X, km.labels_, metric='cosine'), 3),
          'Hierarchical silhouette:', round(silhouette_score(X, ag, metric='cosine'), 3))
    frame['cluster'] = km.labels_; frame.to_csv(OUT / 'news_clusters.csv', index=False)
    joblib.dump(dict(vectorizer=tfidf, model=km), OUT / 'news_kmeans.joblib')
    projection = TruncatedSVD(n_components=2, random_state=42).fit_transform(X)
    plt.figure(); plt.scatter(projection[:, 0], projection[:, 1], c=km.labels_, cmap='viridis')
    plt.title('TF-IDF news clusters (SVD projection)'); plt.tight_layout()
    plt.savefig(OUT / 'news_clusters.png'); plt.close()

if __name__ == '__main__':
    customers(); news()
    print('Datasets: SYNTHETIC instructional examples, not real customers or published news.')
