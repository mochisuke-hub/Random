import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("a")

# テキストファイルから input_prefecture を読み込む
input_path = 'パス'
with open(input_path, 'r', encoding='utf-8') as file:
    input_prefecture_lines = file.readlines()
    input_prefecture_lines = [item.replace('\n', '').replace("\ufeff", "") for item in input_prefecture_lines]

print("b")

# テキストファイルから correct_prefecture を読み込む
correct_path = 'パス'
with open(correct_path, 'r', encoding='utf-8') as file:
    correct_prefecture_lines = file.readlines()

print("c")

# DataFrame の作成
df = pd.DataFrame({'input_prefecture': input_prefecture_lines,
                   'correct_prefecture': correct_prefecture_lines})

print("d")

# TF-IDFベクトライザーを使った特徴量の設計
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4))
X_train_tfidf = vectorizer.fit_transform(df['input_prefecture'])

print("e")

# ランダムフォレストモデルの学習
model = RandomForestClassifier()
model.fit(X_train_tfidf, df['correct_prefecture'])

print("f")

# ユーザーからの入力の受け取り
user_input = input('都道府県名を入力してください: ')

# ユーザーからの入力に対する予測
user_input_tfidf = vectorizer.transform([user_input])
predicted_prefecture = model.predict(user_input_tfidf)

# 修正された都道府県の表示
print(f'入力した都道府県: {user_input}')
print(f'修正された都道府県: {predicted_prefecture[0]}')
