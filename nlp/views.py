from django.shortcuts import render
import fasttext
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# 前処理
# 以下の手順でnameデータのクリーニングを行う.
# アルファベット以外の文字をスペースに置き換える
# ストップワードは削除する
# ステミング（単語の語幹を取り出す作業のこと. 派生語を同じ単語として扱えるようにする）
nltk.download('stopwords') # ストップワード
stopwords = stopwords.words('english')
stemmer = PorterStemmer()
def cleaning(text):
    # アルファベット以外をスペースに置き換え、スペースで分割して単語をリスト化
    words = re.sub(r'[^a-zA-Z]', ' ', text).split()
    # ストップワードは削除してステミング
    stemmed_words = [stemmer.stem(word) for word in words if word not in stopwords]
    # 単語同士をスペースでつなぎ, 文章に戻す
    clean_text = ' '.join(stemmed_words)
    return clean_text

# モデルの読み込み
model = fasttext.load_model("model.bin")

# 予測
def predict(name):
    cleaned_name = cleaning(name)
    categories, probs = model.predict(cleaned_name, k=3)
    return categories, probs

def index(request):
    if request.method == "GET":
        return render(
            request,
            'nlp/home.html'
        )
    else:
        title = request.POST["title"]
        categories, probs = predict(title)
        return render(
            request,
            'nlp/home.html',
            {
                "Product name":title,
                "category1":categories[0].replace('__label__', ''),
                "category2":categories[1].replace('__label__', ''),
                "category3":categories[2].replace('__label__', ''), 
                "prob1":{round(probs[0]*100)},
                "prob2":{round(probs[0]*100)},
                "prob3":{round(probs[0]*100)}
            }
            )