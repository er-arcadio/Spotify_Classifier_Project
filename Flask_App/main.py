from flask import Flask, request, render_template
from make_prediction import top100

# create a flask object
app = Flask(__name__)

# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('index.html')

# creates an association between the /predict_recipe page and the render_message function
# (includes POST requests which allow users to enter in data via form)
@app.route('/predict_top100/', methods=['GET', 'POST'])
def render_message():

    # user-entered ingredients
    features = ['duration_min', 'acoustic', 'danceable', 'energy',
                   'tempo', 'talky', 'happiness', 'loud', 'live', 'instrumental',
                   'mode', 'key']

    # error messages to ensure correct units of measure
    messages = ["The duration must be in minutes.",
                "Don't leave 'acoustic' blank",
                "Something went wrong 'danceable'",
                "Something went wrong 'energy'",
                "Tempo should be in bpm.",
                "Something went wrong 'speechiness'",
                "Something went wrong 'happiness'",
                "Something went wrong 'loudness'",
                "Don't leave 'live' blank",
                "Don't leave 'instrumental' blank",
                "Don't leave 'mode' blank",
                "Invalid 'key' entry (0-11 only)"]

    # hold all amounts as floats
    amounts = []

    # takes user input and ensures it can be turned into a floats
    for i, entry in enumerate(features):
        try:
            user_input = request.form[entry]
            fixed_entry = float(user_input)
        except:
            return render_template('index.html', message=messages[i])

        amounts.append(fixed_entry)

    # show user final message
    final_message = top100(amounts)
    return render_template('index.html', message=final_message)

if __name__ == '__main__':
    app.run(debug=True)