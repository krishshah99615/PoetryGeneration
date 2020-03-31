from flask import Flask, request, render_template

from predict import pre

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        s = str(request.form['line'])
        l= pre(s)
        
        return render_template('index.html',poem=l)
    return render_template('index.html',poem=['Hey write something'])

if __name__ == '__main__':
    app.run(debug=True)