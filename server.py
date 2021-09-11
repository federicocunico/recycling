import json
from typing import Optional
from core import WasteCalendar
from datetime import datetime
from shared import APP_PORT
from flask import Flask, request, Response

app = Flask(__name__)

model: Optional[WasteCalendar] = None

@app.route("/")
def ping():
    today = datetime.today()
    datetime_str = today.strftime("%Y/%m/%d")
    return Response(200, f"System is up! My datetime is: {datetime_str}")

@app.route("/set/<year>")
def set_year(year: str):
    try:
        year = int(year)
    except ValueError:
        return Response(500, f"Unable to convert year {year} to integer.")
    
    if model:
        model.set(year)

@app.route("/get")
def get_today():
    response = model.get_data()
    response_str = json.dumps(response)
    return response_str

@app.route("/get/<year>/<month>/<day>")
def get_month_and_day(year, month, day):
    try:
        requested_str = f"{year}-{month}-{day}"
        requested = datetime.strptime(requested_str, '%Y-%m-%d')
    except:
        return Response(500, f"Unable to understand requested datetime format: {requested_str} (expected year - month - day)")
    
    response = model.get_data(requested)
    response_str = json.dumps(response)
    return response_str

def main():
    global model
    year = datetime.today().year

    model = WasteCalendar(year)

    app.run("0.0.0.0", APP_PORT)

if __name__ == "__main__":
    main()