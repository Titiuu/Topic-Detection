import filesReader
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def transform(dataset, n_features=5000):
    vectorizer = TfidfVectorizer(analyzer=filesReader.split_into_words, max_features=n_features)
    X = vectorizer.fit_transform(dataset)
    return X, vectorizer


def train(X, vectorizer, true_k=14, showLable=False):
    km = KMeans(n_clusters=true_k, max_iter=300, n_init=1)
    km.fit(X)
    if showLable:
        print('Top terms per cluster:')
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster:", i, end='')
            for ind in order_centroids[i, :10]:
                print(terms[ind] + ' ', end='')
            print()
    result = list(km.predict(X))
    print('Cluster distribution:')
    print(dict([(i, result.count(i)) for i in result]))
    return km.inertia_


def test():
    # 测试选择最优参数
    dataset = filesReader.contents('./answer')
    # print("documents:", len(dataset))
    X, vectorizer = transform(dataset, n_features=5000)
    true_ks = []
    scores = []
    for i in range(10, 21):
        score = train(X, vectorizer, true_k=i)
        print(i, score)
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(8, 4))
    plt.plot(true_ks, scores, label="distortion", color="red", linewidth=1)
    plt.xlabel("number of clusters")
    plt.ylabel("dsitortion")
    plt.legend()
    plt.show()


def out():
    # 在最优参数下输出聚类结果
    dataset = filesReader.contents('./answer')
    X, vectorizer = transform(dataset, n_features=5000)
    score = train(X, vectorizer, true_k=14, showLable=True) / len(dataset)
    print(score)


# test()
out()
