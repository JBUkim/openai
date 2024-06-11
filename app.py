import json

import tool

from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.get('/')
def lunch_form():
    meal_options = ['식사', '요리', '간식']
    cuisine_options = ['한식', '중식', '일식', '양식', '아시아']
    situation_options = ['혼밥', '친구', '연인', '가족', '모임']
    
    return render_template('lunch.html', meal_options=meal_options, cuisine_options=cuisine_options, situation_options=situation_options)



# http://localhost:5000/lunch
@app.post('/')
def lunch_proc():
    try:
        data = request.json
        lunch = data['lunch']
        
        lunch = lunch.split(',')
        print('-> lunch:', lunch)
        
        lunch = list(map(int, lunch))
        print('-> lunch:', lunch)

        meal_types = ['식사', '요리', '간식']
        cuisines = ['한국', '중국', '일본', '서양', '아시아']
        situations = ['혼자', '친구와', '연인과', '가족과', '모임에서']
        
        # 사용자가 선택한 카테고리 정보를 조합하여 프롬프트 생성
        selected_meal_types = [meal_types[index] for index, value in enumerate(lunch[:3]) if value == 1]
        selected_cuisines = [cuisines[index] for index, value in enumerate(lunch[3:8]) if value == 1]
        selected_situation = [situations[index] for index, value in enumerate(lunch[8:13]) if value == 1]
        selected_situation_str = selected_situation[0] if selected_situation else ''  # 선택된 상황이 없는 경우를 처리
        
        prompt = f'내가 지금 먹고 싶은건 {", ".join(selected_meal_types)} 이야. 그리고 이 음식은 {", ".join(selected_cuisines)} 음식중에서 골라줘. '
        prompt += f'{"지금 상황은 " + selected_situation_str + " 먹여야 하는 상황이야." if selected_situation_str else ""} '
        prompt += '위의 상황들을 종합해서 메뉴 1개를 추천해줘.'
        prompt += '추천한 메뉴의 레시피를 5줄로 출력해줘.'

        print('-> prompt:', prompt)
        
        format = '''{"res":"음식 추천", "recipe":"레시피 내용"}'''  # 레시피 내용은 실제 레시피로 대체되어야 합니다.

        response = tool.answer('너는 음식 추천 시스템이야.', prompt, format)
        print('-> response:', response)
        
        return response
    except Exception as e:
        print('Error:', str(e))
        return {'error': str(e)}, 500
