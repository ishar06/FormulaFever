# app.py
from flask import Flask, render_template
import requests

app = Flask(__name__)

@app.route('/')
def home():
    # Call the Jolpica (Ergast replacement) API
    url = "https://api.jolpi.ca/ergast/f1/current/driverStandings.json"
    
    try:
        data = requests.get(url).json()

        # Dig into the JSON to get the list of drivers
        standings_list = data['MRData']['StandingsTable']['StandingsLists']
        
        if not standings_list:
            return render_template('home.html', standings=None, error="No standings data found.")

        standings = standings_list[0]['DriverStandings']
        season = standings_list[0]['season']

        return render_template('home.html', standings=standings, season=season)
    
    except requests.exceptions.RequestException as e:
        return render_template('home.html', standings=None, error=f"Could not connect to API: {e}")
    except KeyError:
        return render_template('home.html', standings=None, error="Error parsing data from API.")


if __name__ == '__main__':
    app.run(debug=True)