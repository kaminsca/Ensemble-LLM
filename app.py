from flask import Flask, render_template, request, redirect  # Add this line to import 'redirect'
import torch
from main1 import RNN, generateDict, create_tensors, load_data

app = Flask(__name__)

model = torch.load('output/model.torch')  # Load the pre-trained model
word2idx = generateDict([])  # You can provide an empty list as a placeholder

class_labels = {
    0: 'CONV',
    1: 'SCIENCE',
    2: 'MATH',
    3: 'LAW',
    4: 'CODE',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        question = request.form['question']
        train_data_raw, _ = load_data('train')
        word2idx = generateDict(train_data_raw)
        question_tensor = create_tensors([question], word2idx)[0]

        print("Question Tensor:", question_tensor)  # Debug statement

        with torch.no_grad():
            out = model(question_tensor)

            print("Raw Output from Model:", out)  # Debug statement

        pred_label = torch.argmax(out).item()
        pred_class = class_labels[pred_label]

        print("Predicted Label:", pred_label)  # Debug statement
        print("Predicted Class:", pred_class)  # Debug statement

        # Determine the hyperlink based on the predicted class
        if pred_class == 'CONV':
            return redirect("https://chat.openai.com/")
        elif pred_class == 'MATH':
            return redirect("https://www.tutoreva.com/?gclid=Cj0KCQiAyeWrBhDDARIsAGP1mWRoC-3c_6P4Lfp2UQ2YXdZOdcwiPyqSjHj0hn48p3txplynAzH2BqIaAqKbEALw_wcB")
        elif pred_class == 'LAW':
            return redirect("https://www.chatlaw.us/")
        elif pred_class == 'CODE':
            return redirect("https://github.com/features/copilot")
        elif pred_class == 'Logic':
            return redirect("https://gpt3demo.com/apps/roberta#google_vignette")
        else:
            # Handle other classes or provide a default redirect
            return redirect("/")

        # Uncomment the following line if you want to render a template instead
        # return render_template('result.html', question=question, prediction=pred_class)

if __name__ == '__main__':
    app.run(debug=True)
