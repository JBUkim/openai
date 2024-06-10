from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    filenames = [i for i in range(1, 26)] # 1 ~ 25
    
    return render_template('./index.html', filenames=filenames) # html template으로 데이터 전달