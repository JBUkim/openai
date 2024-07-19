import os
import re
import sys
import time
import threading
import requests
import base64
import csv
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from matplotlib import font_manager, rc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Flask, request, render_template, session, jsonify, Response
from flask_cors import CORS
import cx_Oracle
from konlpy.tag import Hannanum
from cloud import WordCloud  # Assuming 'cloud' refers to the 'wordcloud' library
import tool
from llama_index.core import StorageContext, load_index_from_storage
import logging

api_key=os.getenv('OPENAI_API_KEY')

def create_thumbnail(image_path, thumbnail_path, width, height):
    with Image.open(image_path) as img:
        img.thumbnail((width, height))
        img.save(thumbnail_path, "JPEG")


# Function to encode the image
def encode_image(image_path, image_file):
    with image_file as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

app = Flask(__name__)

app.secret_key = 'secret_key'

CORS(app)

@app.get('/lunch')
def lunch_form():
    # 세션 불러오기
    accountno = request.args.get('accountno')
    print(accountno)
    session['accountno'] = accountno
    
    meal_options = ['식사', '요리', '간식']
    cuisine_options = ['한식', '중식', '일식', '양식', '아시아']
    situation_options = ['혼밥', '친구', '연인', '가족', '모임']
    
    return render_template('index.html', meal_options=meal_options, cuisine_options=cuisine_options, situation_options=situation_options)

@app.post('/lunch')
def lunch_proc():
    accountno = 0
    accountno = session.get('accountno', accountno)
    print(accountno)
    
    os.putenv('NLS_LANG', 'KOREAN_KOREA.KO16MSWIN949')

    conn = cx_Oracle.connect('team3/69017000@44.205.155.56:1521/XE')
    cs = conn.cursor()

    try:
        data = request.json
        lunch = data['lunch']
        
        lunch = lunch.split(',')
        print('-> lunch:', lunch)
        
        lunch = list(map(int, lunch))
        print('-> lunch:', lunch)

        meal_types = ['식사', '요리', '간식']
        cuisines = ['한국 음식', '중국집 음식', '일본음식', '서양식', '태국, 베트남 음식']
        situations = ['혼자', '친구와', '연인과', '가족과', '모임에서']
        
        # 사용자가 선택한 카테고리 정보를 조합하여 프롬프트 생성
        selected_meal_types = [meal_types[index] for index, value in enumerate(lunch[:3]) if value == 1]
        selected_cuisines = [cuisines[index] for index, value in enumerate(lunch[3:8]) if value == 1]
        selected_situation = [situations[index] for index, value in enumerate(lunch[8:13]) if value == 1]
        selected_situation_str = selected_situation[0] if selected_situation else ''  # 선택된 상황이 없는 경우를 처리
        
        
        prompt = f'{"지금 상황은 " + selected_situation_str + " 음식을 먹을꺼야." if selected_situation_str else ""} '
        prompt += f'내가 지금 먹고 싶은건 {", ".join(selected_meal_types)} 종류이다. 이 음식은 {", ".join(selected_cuisines)} 20가지 리스트 중 랜덤으로 한개를 골라줘.'
        prompt += '추천한 음식의 레시피를 5줄로 출력해줘.'

        print('-> prompt:', prompt)
        
        format = '''
        {"res":"음식 추천", 
        "recipe":"레시피 내용"
        }
        '''  # 레시피 내용은 실제 레시피로 대체되어야 합니다.
        
        response = tool.answer('너는 음식 추천 시스템이야.', prompt, format)
        print('-> response:', response)

        # 응답을 JSON으로 파싱
        resp = jsonify({'res': response['res'], 'recipe': response['recipe']})
        
        cs.execute("""
            INSERT INTO L_RECOM (L_NO, ACCOUNTNO, L_MENU, L_RECIPE, L_DATE)
            VALUES (L_RECOM_SEQ.nextval, :1, :2, :3, sysdate)
        """, (accountno, response['res'], response['recipe']))
        
        conn.commit()

        return resp
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cs.close()
        conn.close()
        
# ------------------------------------------------------------------------------------------------
# Chatbot 관련
# ------------------------------------------------------------------------------------------------

# 로그 레벨 설정
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, force=True)

# 인덱스 로드
storage_context = StorageContext.from_defaults(persist_dir="./storage") # /openai/llama_index/storage 폴더 생성
index = load_index_from_storage(storage_context)

# 쿼리 엔진 생성
query_engine = index.as_query_engine()

@app.get("/chatbot") # http://localhost:5000/chatbot
def chatbot_form():
    return render_template("chatbot.html")

@app.post("/chatbot") # http://localhost:5000/chatbot
def chatbot_proc():
    data = request.json
    # f=request.files['file']

    question = data['question']
    print('-> question:', question)
    # return;

    response = query_engine.query(question)
    
    obj = {
        "res": response.response
    }
    
    return jsonify(obj) # dictionary -> json + HTTP 응답 객체

# ------------------------------------------------------------------------------------------------

# 허용 가능한 파일 확장자 설정 (예: 이미지 파일만 허용하도록 설정)
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 25 M 제한
def allowed_size(size):
    return True if size <= 1024 * 1024 * 25 else False

# Ajax 기반 파일 업로드 폼    
@app.get("/menu_web") # http://localhost:5000/menu_web
def menu_web_form():
    accountno = request.args.get('accountno')
    managerno = request.args.get('managerno')
    session['accountno'] = accountno
    print(managerno)
    print(accountno)
    # accountno만 존재하는 경우
    if accountno:
        return render_template("menu_web.html")
    # managerno만 존재하는 경우
    elif managerno:
        return render_template("menu_web.html")
    # accountno와 managerno 모두 존재하지 않는 경우
    else:
        return redirect("http://15.165.140.165:9093/answer/list")

# Ajax 기반 파일 업로드 처리    
@app.post("/menu_web") # http://localhost:5000/menu_web
def menu_web_proc():
    # data = request.json
    # article = data["article"]
    accountno = 0
    accountno = session.get('accountno', accountno)
    print(accountno)

    f=request.files['file']
    file_size = len(f.read())
    f.seek(0) # 파일 포인터를 처음으로 이동
    
    # print('-> file_size:', file_size)
    
    if allowed_size(file_size) == False: # 25 MB 초과
        resp = jsonify({'message': "파일 사이즈가 25M를 넘습니다. 파일 용량: " + str(round(file_size/1024/1024)) + ' MB'})
        resp.status_code = 500 # 서버 에러
    else: # 25 MB 이하
        if f and allowed_file(f.filename): # 허용 가능한 파일 확장자인지 확인
            # 저장할 경로 지정 (예: 'uploads' 폴더에 저장)
            upload_folder = 'C:\\kd\\deploy\\team3_v2sbm3c\\contents\\storage'
            # 우분투
            # upload_folder = '/home/ubuntu/deploy/team3_v2sbm3c/contents/storage/'
            
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            print('-> f.filename', f.filename)
            f.save(os.path.join(upload_folder, f.filename)) # 파일 저장
            name, ext = os.path.splitext(f.filename)  # 파일명과 확장자를 분리
            thumbnail_filename = f"{name}_t{ext}"   # 새로운 파일명 생성
            thumbnail_path = os.path.join(upload_folder, thumbnail_filename)
            print(upload_folder, thumbnail_path)
            create_thumbnail(os.path.join(upload_folder, f.filename), thumbnail_path, 200, 150)
            
            os.putenv('NLS_LANG', 'KOREAN_KOREA.KO16MSWIN949')

            conn = cx_Oracle.connect('team3/69017000@44.205.155.56:1521/XE')
            cs = conn.cursor()
            
            cs.execute("""
                    INSERT INTO ai_search (searchno, img_search, img_search_save, img_search_thumb, img_search_size, accountno, answerno, rdate)
                    VALUES (ai_search_seq.nextval, :1, :2, :3, :4, :5, ai_answer_seq.nextval, sysdate)
                    """, (f.filename, f.filename, thumbnail_filename, file_size, accountno))
            
            cs.execute("SELECT ai_search_seq.CURRVAL FROM DUAL")
            recent_searchno = cs.fetchone()[0]
            
            cs.execute("SELECT ai_answer_seq.CURRVAL FROM DUAL")
            recent_answerno = cs.fetchone()[0]
            
            # return "파일을 전송했습니다."

            image_path = os.path.join(upload_folder, f.filename)
            
            # Getting the base64 string
            image_file = open(image_path, "rb")
            base64_image = encode_image(image_path, image_file)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                }

            prompt  = "메뉴 알려줘"
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                                },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                "max_tokens": 300
                }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            menu = response.json()['choices'][0]['message']['content']
            print(menu)
            
            prompt = menu + '레시피 자세하게 알려줘'
            format = '''
            {
                "res": "메뉴 제작 방법"
                }
                '''

            response = tool.answer('너는 요리사야', prompt, format)
            print('-' * 80)
            print("레시피 제작 방법")
            print(response['res'])
            resp = jsonify({'message': response['res']}) # dict -> json string
            
            cs.execute("""
                    INSERT INTO ai_answer (answerno, text_answer, searchno, accountno)
                    VALUES (:1, :2, :3, :4)
                    """, (recent_answerno, response['res'], recent_searchno, accountno))

            resp.status_code = 201 # 정상 처리

        else:
            resp = jsonify({'message': '전송 할 수 없는 파일 형식입니다.'})
            # print(resp)
            resp.status_code = 500 # 서버 에러

        # upload된 파일 삭제 
        # 파일이 존재하는지 확인 후 삭제 
        
        cs.close()
        conn.commit()
        conn.close()
        
        image_file.close() # PermissionError 발생함으로 파일을 닫을 것.
        f.close()

        # print('-> os.path.join(upload_folder, f.filename): ', os.path.join(upload_folder, f.filename))
        # # 삭제할 파일 경로 조합, storage\진주_조개잡이_쿨.mp3
        # delete_file = os.path.join(upload_folder, f.filename) 

        # if os.path.exists(delete_file):
        #     os.remove(delete_file) # 파일 이용을 모두 했음으로 파일 삭제
        #     # os.remove('./storage/진주_조개잡이_쿨.mp3')
        #     print(f'{delete_file} 파일 삭제')
        # else:
        #     print(f'{delete_file} 파일 삭제 실패')

    
    return resp

# ------------------------------------------------------------------------------------------------
## 데이터 분석 그래프 최신화
# ------------------------------------------------------------------------------------------------
# Oracle DB 연결 설정
conn = cx_Oracle.connect('team3/69017000@44.205.155.56:1521/XE')
cs = conn.cursor()

# 데이터베이스에서 데이터 가져오기 함수
def fetch_data():
    query = """
        SELECT 
            r.SPICENO,
            s.SPICENAME,
            COUNT(r.SPICENO) AS cnt
        FROM 
            RECOMMEND r
        JOIN 
            SPICE s ON r.SPICENO = s.SPICENO
        GROUP BY 
            r.SPICENO, s.SPICENAME
        """
    
    cs.execute(query)
    return cs.fetchall()

# CSV 파일로 데이터 저장 함수
def save_to_csv(data, filename='recommend_data.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['SPICENO', 'SPICENAME', 'COUNT'])
        writer.writerows(data)

# 실시간 그래프 업데이트 함수
def generate_graph():
    while True:
        data = fetch_data()

        nos = [row[0] for row in data]
        names = [row[1] for row in data]
        counts = [row[2] for row in data]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, counts)
        ax.set_xlabel('향신료 이름')
        ax.set_ylabel('추천 횟수')
        ax.set_title('향신료 추천')

        # 그래프 이미지를 바이너리로 변환
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        plt.close(fig)

        yield (b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + output.getvalue() + b'\r\n')

        # 데이터를 CSV 파일로 저장
        save_to_csv(data)

        time.sleep(1)  # 1초마다 업데이트

# ------------------------------------------------------------------------------------------------
        
# Matplotlib 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# ------------------------------------------------------------------------------------------------

# 데이터베이스에서 BCONTENT 필드의 데이터를 가져오는 함수
def cloud_data():
    cs.execute("SELECT BCONTENT FROM BOARD")
    return cs.fetchall()

# 텍스트에서 한글 단어만 추출하는 함수
def process_text(text):
    words = re.findall(r'\b[가-힣]+\b', text)
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

    icon = Image.open('static\img\cloud.png')
    img = Image.new('RGB', icon.size, (255, 255, 255))
    img.paste(icon, icon)
    img = np.array(img)

    if platform.system() == 'Windows':
        font_path = "C:/Windows/Fonts/malgun.ttf"
    elif platform.system() == "Darwin":
        font_path = "/Users/$USER/Library/Fonts/AppleGothic.ttf"

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
    plt.savefig('./static/bcontent.png')  # 워드클라우드를 static 폴더에 저장
    plt.close()
    
# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# 검색어 데이터 분석
def search_data():
    conn = cx_Oracle.connect('team3/69017000@44.205.155.56:1521/XE')
    cs = conn.cursor()
    cs.execute("SELECT word, cnt FROM search")
    return cs.fetchall()

def update_graph():
    try:
        previous_data = None

        while True:
            data = search_data()
            sorted_data = sorted(data, key=lambda x: x[1], reverse=True)[:5]
            
            if sorted_data != previous_data:
                words = [row[0] for row in sorted_data]
                counts = [row[1] for row in sorted_data]
                
                with open('recipe.csv', 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['WORD', 'CNT'])
                    csvwriter.writerows(sorted_data)
                
                with open('recipe.csv', 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    next(csvreader)
                    data_from_csv = list(csvreader)
                    
                words = [row[0] for row in data_from_csv]
                counts = [int(row[1]) for row in data_from_csv]
                
                f = plt.figure(figsize=(10, 5))
                bars = plt.bar(words, counts, color='skyblue')
                
                plt.xlabel('검색한 레시피', fontsize=14)
                plt.ylabel('검색 횟수', fontsize=14)
                plt.title('검색 순위', fontsize=16)
                
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
                                ha='center', va='bottom', fontsize=10)
                    
                plt.ylim(0, max(counts) * 1.1)  # y축의 최대값을 기존 최대값의 110%로 설정
                
                plt.savefig("C:\\kd\\ws_java\\team3_v2sbm3c\\src\\main\\resources\\static\\images\\graph1.png")
                plt.close(f)
                
                previous_data = data

            time.sleep(1)

    except KeyboardInterrupt:
        print("실시간 그래프 업데이트가 중지되었습니다.")

    finally:
        cs.close()
        conn.close()

# ------------------------------------------------------------------------------------------------

@app.route('/graph')
def graph():
    return Response(generate_graph(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/wordcloud')
def wordcloud():
    create_wordcloud()
    return render_template('wordcloud.html')

# Flask 서버 시작
if __name__ == '__main__':
    graph_thread = threading.Thread(target=update_graph)
    graph_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=True)  # 0.0.0.0: 모든 Host 에서 접속 가능