import platform
from flask import Flask, request, render_template, session, jsonify, Response
from flask_cors import CORS
import cx_Oracle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import re
from collections import Counter
import numpy as np
from konlpy.tag import Hannanum
import pandas as pd
from wordcloud import WordCloud
import os

app = Flask(__name__)
app.secret_key = 'secret_key'
CORS(app)

# 데이터베이스에서 BCONTENT 필드의 데이터를 가져오는 함수
def cloud_data():
    conn = cx_Oracle.connect('team3/69017000@44.205.155.56:1521/XE')
    cs = conn.cursor()
    cs.execute("SELECT BCONTENT FROM BOARD")
    return cs.fetchall()

# 텍스트에서 한글 단어만 추출하는 함수
def process_text(text):
    words = re.findall(r'[가-힣]+', text)
    return words

# 워드클라우드 생성 함수
def create_wordcloud():
    data = cloud_data()
    content_list = [row[0].read() for row in data]
    all_content = ' '.join(content_list)
    processed_content = re.sub('[^가-힣]', ' ', all_content)

    words = process_text(processed_content)
    word_counts = Counter(words)

    han = Hannanum()
    nouns = han.nouns(processed_content)

    STOPWORDS = []
    new_nouns = [item for item in nouns if item not in STOPWORDS]
    nouns = new_nouns

    df = pd.DataFrame({'word': nouns})
    df['len'] = df['word'].str.len()
    df = df.query('len >= 2')
    df = df.sort_values(by=['len'], ascending=True)

    df2 = df.groupby(['word'], as_index=False).agg(n=('word', 'count')).sort_values(['n'], ascending=False)
    top100 = df2.reset_index(drop=True).head(100)
    top100 = top100.set_index('word')
    dict_df = top100.to_dict()['n']

    icon_path = os.path.join('static', 'img', 'cloud.png')
    icon = Image.open(icon_path)
    img = Image.new('RGB', icon.size, (255, 255, 255))
    img.paste(icon, icon)
    img = np.array(img)

    font_path = "C:/Windows/Fonts/malgun.ttf" if platform.system() == 'Windows' else "/Users/$USER/Library/Fonts/AppleGothic.ttf"

    wc = WordCloud(random_state=1234, 
                    font_path=font_path, 
                    width=800,
                    height=400,
                    background_color='white',
                    mask=img)

    wc_img = wc.generate_from_frequencies(dict_df)

    plt.figure(figsize=(15, 10))
    plt.axis('off')
    plt.imshow(wc_img)
    output_path = os.path.join('static', 'bcontent.png')
    plt.savefig(output_path)  # 워드클라우드를 static 폴더에 저장
    plt.close()

@app.route('/wordcloud')
def wordcloud():
    create_wordcloud()
    return render_template('wordcloud.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
