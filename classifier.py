import filesReader
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 读取文件内容，返回内容列表和标签列表
X, y = filesReader.contents_and_labels('./answer')

# 划分训测集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# Tfidf提取特征，向量化
vectorizer = TfidfVectorizer(analyzer=filesReader.split_into_words, encoding='gb18030', max_features=5000)
X_train_fit = vectorizer.fit(X)
X_train_vector = vectorizer.transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# 朴素贝叶斯分类器(多项式模型)
nb_clf = MultinomialNB()
nb_clf = nb_clf.fit(X_train_vector, y_train)
y_nbpred = nb_clf.predict(X_test_vector)
print('朴素贝叶斯准确率: {:.1%}'.format(accuracy_score(y_test, y_nbpred)))

# 随机森林分类器
forest_clf = RandomForestClassifier()
forest_clf = forest_clf.fit(X_train_vector, y_train)
y_forestpred = forest_clf.predict(X_test_vector)
print('随机森林准确率: {:.1%}'.format(accuracy_score(y_test, y_forestpred)))

# 支持向量机分类器
sv_clf = svm.SVC(kernel='linear')
sv_clf = sv_clf.fit(X_train_vector, y_train)
y_svpred = sv_clf.predict(X_test_vector)
print('支持向量机准确率: {:.1%}'.format(accuracy_score(y_test, y_svpred)))

# Boosting模型分类器
xgb_clf = XGBClassifier()
xgb_clf = xgb_clf.fit(X_train_vector, y_train)
y_xgbpred = xgb_clf.predict(X_test_vector)
print('Boosting模型准确率: {:.1%}'.format(accuracy_score(y_test, y_xgbpred)))

