import json
import os
from flask import Flask, request, render_template, session, jsonify
from flask_cors import CORS
import cx_Oracle
import tool

app = Flask(__name__)

app.secret_key = 'secret_key'

CORS(app)

@app.get('/')
def lunch_form():
    # 세션 불러오기
    accountno = request.args.get('accountno')
    print(accountno)
    session['accountno'] = accountno
    
    meal_options = ['식사', '요리', '간식']
    cuisine_options = ['한식', '중식', '일식', '양식', '아시아']
    situation_options = ['혼밥', '친구', '연인', '가족', '모임']
    
    return render_template('index.html', meal_options=meal_options, cuisine_options=cuisine_options, situation_options=situation_options)

@app.post('/')
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
        cuisines = ['한국의 유명한 20가지', '중국의 유명한 20가지', '일본의 유명한 20가지', '양식당에서 유명한 20가지', '태국이나 베트남에서 유명한 20가지']
        situations = ['혼자', '친구와', '연인과', '가족과', '모임에서']
        
        # 사용자가 선택한 카테고리 정보를 조합하여 프롬프트 생성
        selected_meal_types = [meal_types[index] for index, value in enumerate(lunch[:3]) if value == 1]
        selected_cuisines = [cuisines[index] for index, value in enumerate(lunch[3:8]) if value == 1]
        selected_situation = [situations[index] for index, value in enumerate(lunch[8:13]) if value == 1]
        selected_situation_str = selected_situation[0] if selected_situation else ''  # 선택된 상황이 없는 경우를 처리
        
        
        prompt = f'{"지금 상황은 " + selected_situation_str + " 먹여야 하는 상황이야." if selected_situation_str else ""} '
        prompt += f'내가 지금 먹고 싶은건 {", ".join(selected_meal_types)} 이야. 그리고 이 음식은 {", ".join(selected_cuisines)} 음식중에서 한가지만 골라줘. '
        prompt += '위의 상황들을 종합해서 맛있는 음식 1개를 추천해줘.'
        prompt += '추천한 메뉴의 레시피를 5줄로 출력해줘.'

        print('-> prompt:', prompt)
        
        format = '''{"res":"음식 추천", "recipe":"레시피 내용"}'''  # 레시피 내용은 실제 레시피로 대체되어야 합니다.
        
        response = tool.answer('너는 음식 추천 시스템이야.', prompt, format)
        print('-> response:', response)

        # 응답을 JSON으로 파싱
        response_data = json.loads(response)
        
        cs.execute("""
            INSERT INTO L_RECOM (L_NO, ACCOUNTNO, L_MENU, L_RECIPE, L_DATE)
            VALUES (L_RECOM_SEQ.nextval, :1, :2, :3, sysdate)
        """, (accountno, response_data['res'], response_data['recipe']))
        
        conn.commit()

        return jsonify(response_data)
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        cs.close()
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
