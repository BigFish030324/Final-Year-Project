from flask import Flask, render_template

# ----------------------------------------------------
)

app = Flask(__name__)

    return render_template(
        error = error,

@app.route('/')
def website():
    return render_template('index.html')

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )

# Run this block of code in terminal
# flask --app main run --host 127.0.0.1 --port 8000 --debug
