import sys
from flask import Flask, render_template, request
sys.path.append('app/ml')
from lstm import run_model

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            from_year = int(request.form.get("from_year"))
            ref_days = int(request.form.get("ref_days"))
            code = request.form.get("code")
            plot_url, test_score, company_name = run_model(from_year, ref_days, code)
            return render_template("plot.html", plot_url=plot_url, test_score=test_score, company_name=company_name)
        except ValueError as e:
            return render_template("index.html", error=str(e))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)