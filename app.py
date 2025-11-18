from flask import Flask, render_template, request
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ✅ Load real API key from .env file


genai.configure(api_key="AIzaSyAEACKU_z_SkUpHISQAjPeNOeJBtm_TwJc")

# ✅ Use publicly accessible model
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_manual(car_model, usage):
    prompt = f"""
    You are a smart automotive assistant.
    Generate a personalized car manual for the following:

    Car Model: {car_model}
    Usage Pattern: {usage}

    Include:
    1. Top 3 Maintenance Tips
    2. 2 Common Repair Guides
    3. 5 Most Likely FAQs and Answers
    Make it clear and easy to understand.
    """
    response = model.generate_content(prompt)
    return response.text

@app.route("/", methods=["GET", "POST"])
def index():
    manual = None
    if request.method == "POST":
        car_model = request.form["car_model"]
        usage = request.form["usage"]
        manual = generate_manual(car_model, usage)
    return render_template("index.html", manual=manual)

if __name__ == "__main__":
    app.run(debug=True)
