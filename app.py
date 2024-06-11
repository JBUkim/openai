import json

from flask import Flask, request, render_template
from flask_cors import CORS

import tool

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    filenames = [i for i in range(1, 26)] # 1 ~ 25
    
    return render_template('./index.html', filenames=filenames) # html template으로 데이터 전달

# http://localhost:5000/
@app.post('/')
def movie_proc():
    data = request.json
    movie= data['movie']
    
    movie= movie.split(',')
    print('-> movie:', movie)
    
    # movie 배열의 요소를 정수로 변경하여 list로 변경
    movie= list(map(int, movie)) # map: 배열의 요소에 함수를 적용하는 기능을 함.
    print('-> movie:', movie)

    # return movie
    
    movies = ['반지의 제왕', 'A Quiet Place (2018)', '러브액츄얼리', '화이트 칙스', 'Interstellar (2014)',
            '해리포터와 마법사의 돌', 'The Autopsy of Jane Doe (2016)', '타이타닉', '세 얼간이', 'A.I. (2001)',
            '캐리비안의 해적', 'The Conjuring (2013)', '맘마미아', '덤 앤 더머', 'The Martian (2015)',
            '닥터 스트레인지', 'The Exorcist (1973)', 'La La Land (2016)', '우리는 동물원을 샀다.', 'Edge of Tomorrow (2014)',
            '아바타 (2009)', 'The Rite (2011)', '비긴 어게인', '미트 페어런츠', 'Gravity (2013)']
    
    watch = []
    for index in range(len(movie)): # 0 ~ 24
        if movie[index] == 1: # 시청한 영화이면
            watch.append(movies[index]) # 시청한 영화의 이름을 추가

    watch_join = ','.join(watch) # 시청한 영화를 ","로 구분
    
    print('->watch_join:', watch_join)
    # return watch_join  # 콰이어트 플레이스,제인도,컨저링,엑소시스트,더 라이트

    prompt = f'내가 시청한 영화는 [{watch_join}]이야, 내가 시청한 장르의 영화 중 2000년 이후에 출시되고, 내가 시청하지 않고 평점이 높은 영화 5편을 추천해줘.'
    print('-> prompt:', prompt)
    
    format = '''{"res":"영화 목록"}'''

    response = tool.answer('너는 영화 추천 시스템이야.', prompt, format)
    print('-> response:', response)
    
    return response  # json 객체 전달