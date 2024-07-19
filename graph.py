
from flask import Flask, request, render_template, session, jsonify, Response
from flask_cors import CORS
import cx_Oracle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import font_manager, rc
import io
import csv
import time

app = Flask(__name__)
app.secret_key = 'secret_key'
CORS(app)


## 데이터 분석 그래프 최신화

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
        
# Matplotlib 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


    
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
            
            # 이전 데이터와 현재 데이터를 비교
            if sorted_data != previous_data:
                words = [row[0] for row in sorted_data]
                counts = [row[1] for row in sorted_data]
                
                # CSV 파일로 저장
                with open('recipe.csv', 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['WORD', 'CNT'])
                    csvwriter.writerows(sorted_data)
                
                # CSV 파일 읽기
                with open('recipe.csv', 'r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    next(csvreader)  # 헤더 건너뛰기
                    data_from_csv = list(csvreader)
                    
                words = [row[0] for row in data_from_csv]
                counts = [int(row[1]) for row in data_from_csv]
                
                f = plt.figure(figsize=(10, 5))
                plt.bar(words, counts)
                plt.xlabel('레시피 이름')
                plt.ylabel('검색 횟수')
                plt.title('가장 많이 찾은 레시피')
                plt.savefig("C:\\kd\\ws_java\\team3_v2sbm3c\\src\\main\\resources\\static\\images\\graph1.png")
                plt.close(f)
                # webbrowser.open_new('graph.html')
                # 현재 데이터를 이전 데이터로 저장
                previous_data = data

            time.sleep(1)  # 1초마다 업데이트

    except KeyboardInterrupt:
        print("실시간 그래프 업데이트가 중지되었습니다.")

    finally:
        cs.close()
        conn.close()

@app.route('/graph')
def graph():
    return Response(generate_graph(), mimetype='multipart/x-mixed-replace; boundary=frame')
  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  # 0.0.0.0: 모든 Host 에서 접속 가능, python llama_chatbot.py