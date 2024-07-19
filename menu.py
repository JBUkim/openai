
import os
import base64
import requests
from PIL import Image
from flask import Flask, request, render_template, session, jsonify, Response
from flask_cors import CORS
import cx_Oracle
import tool

api_key=os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

app.secret_key = 'secret_key'

CORS(app)

def create_thumbnail(image_path, thumbnail_path, width, height):
    with Image.open(image_path) as img:
        img.thumbnail((width, height))
        img.save(thumbnail_path, "JPEG")


# Function to encode the image
def encode_image(image_path, image_file):
    with image_file as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  # 0.0.0.0: 모든 Host 에서 접속 가능, python llama_chatbot.py