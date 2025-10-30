# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import requests
from analysis import create_laptime_chart
import fastf1 as ff1  # <-- NEW: Import FastF1 here

app = Flask(__name__)

# NEW: A secret key is required to use flash messages
app.config['SECRET_KEY'] = 'a_very_secret_key_change_this'

# --- NEW: Enable FastF1 Cache ---
# This is the correct place to do this. 
# It runs once when the app starts.
try:
    ff1.Cache.enable_cache('ff1_cache/')
    print("FastF1 cache enabled.")
except Exception as e:
    print(f"CRITICAL: Could not enable FastF1 cache: {e}")
# --- END NEW ---


@app.route('/')
def home():
    url = "https://api.jolpi.ca/ergast/f1/current/driverStandings.json"
    
    try:
        data = requests.get(url).json()
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

# ----------------------------------------------
# --- UPDATED: ROUTE WITH STRICT VALIDATION ---
# ----------------------------------------------
@app.route('/race_search')
def race_search():
    year = request.args.get('year')
    race_name_input = request.args.get('race_name')

    if not year or not race_name_input:
        flash('Both Year and Race Name are required.', 'danger')
        return redirect(url_for('home'))

    try:
        year_int = int(year)
    except ValueError:
        flash('Year must be a valid number.', 'warning')
        return redirect(url_for('home'))

    try:
        # --- NEW VALIDATION LOGIC ---
        print(f"Validating schedule for {year_int}...")
        # Get the official event schedule for that year
        schedule = ff1.get_event_schedule(year_int, include_testing=False)
        
        # FastF1 is smart, it can match by Location, Country, or Event Name
        # We need to check our input against all three.
        
        # .str.lower() makes the check case-insensitive
        valid_locations = set(schedule['Location'].str.lower())
        valid_countries = set(schedule['Country'].str.lower())
        valid_event_names = set(schedule['EventName'].str.lower())
        
        input_lower = race_name_input.lower()
        
        # This is our strict check
        if (input_lower not in valid_locations and 
            input_lower not in valid_countries and 
            input_lower not in valid_event_names):
            
            # If the input is not in any of those sets, it's invalid.
            flash(f"Error: '{race_name_input}' is not a valid race location, country, or event name for {year_int}.", 'danger')
            return redirect(url_for('home'))
            
        # --- END VALIDATION LOGIC ---

        # If we get here, the name is valid.
        # We .capitalize() to make it "Monza" or "Italy", which FastF1 prefers.
        return redirect(url_for('race_dashboard', year=year_int, race_name=race_name_input.capitalize()))

    except Exception as e:
        # This will catch other errors, e.g., if FastF1 can't get the schedule
        flash(f"An error occurred while validating the race: {e}", 'danger')
        return redirect(url_for('home'))


# ----------------------------------------------
# --- DAY 3 ROUTE (Now more robust) ---
# ----------------------------------------------
@app.route('/race/<int:year>/<string:race_name>')
def race_dashboard(year, race_name):
    # This route now trusts it's getting a valid year and race_name
    # The try/except is for data *processing* errors, not invalid names
    try:
        lap_chart_html, summary = create_laptime_chart(year, race_name)
    
    except Exception as e:
        flash(f"Error processing data for {year} {race_name}. The session might not have lap data available. (Error: {e})", 'danger')
        return redirect(url_for('home'))

    return render_template('race.html', 
                           year=year, 
                           race_name=race_name, 
                           lap_chart_html=lap_chart_html,
                           summary=summary)


if __name__ == '__main__':
    app.run(debug=True)