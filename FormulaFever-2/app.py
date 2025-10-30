import os
from groq import Groq
from flask import Flask, render_template, request, redirect, url_for, flash
import requests
from analysis import create_laptime_chart, create_comparison_plots
import fastf1 as ff1  
from analysis import create_laptime_chart, create_driver_improvement_chart
import datetime

app = Flask(__name__)

app.config['SECRET_KEY'] = 'a_very_secret_key_change_this'

# --- START: NEW Groq Configuration ---
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("CRITICAL: GROQ_API_KEY environment variable not set. "
                     "Please set the key and restart the app.")
try:
    client = Groq(api_key=api_key) # Initialize the Groq client
    print("✅ Groq AI Client initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Could not initialize Groq AI Client. Error: {e}")
    raise e
# --- END: NEW Groq Configuration ---


try:
    ff1.Cache.enable_cache('ff1_cache/')
    print("FastF1 cache enabled.")
except Exception as e:
    print(f"CRITICAL: Could not enable FastF1 cache: {e}")


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








@app.route('/comparator')
def comparator():
    # Renders the new comparator.html file
    return render_template('comparator.html')

# 2. This route HANDLES the form submission from that page
@app.route('/compare', methods=['POST'])
def compare_laps():
    try:
        # 1. Get the data from the form
        year = int(request.form['year'])
        gp = request.form['gp']
        session = request.form['session']
        driver1 = request.form['driver1'].upper()
        driver2 = request.form['driver2'].upper()

        # 2. Call your *new* analysis function
        plot_data = create_comparison_plots(year, gp, session, driver1, driver2)

        # 3. Check for an error from the analysis
        if "error" in plot_data:
            return render_template('comparator.html', error_message=plot_data['error'])
        
        # 4. Render the SAME comparator page, but pass in the plot data
        return render_template('comparator.html', 
                               delta_plot=plot_data['delta_plot'], 
                               telemetry_plot=plot_data['telemetry_plot'])
    
    except Exception as e:
        # Catch any other general errors
        return render_template('comparator.html', error_message=str(e))



# new change : 
@app.route('/driver_search')
def driver_search():
    year = request.args.get('year')
    driver_abbr = request.args.get('driver_abbr')

    if not year or not driver_abbr:
        flash('Year and Driver Abbreviation are required.', 'danger')
        return redirect(url_for('home'))

    try:
        year_int = int(year)
    except ValueError:
        flash('Year must be a valid number.', 'warning')
        return redirect(url_for('home'))
    
    # Basic validation for abbreviation
    if not (len(driver_abbr) == 3 and driver_abbr.isalpha()):
         flash(f"'{driver_abbr}' is not a valid 3-letter abbreviation (e.g., 'VER').", 'warning')
         return redirect(url_for('home'))
    
    # Clean up the abbreviation (to uppercase) and redirect
    driver_abbr_upper = driver_abbr.upper()
    return redirect(url_for('driver_dashboard', year=year_int, driver_abbr=driver_abbr_upper))


@app.route('/driver/<int:year>/<string:driver_abbr>')
def driver_dashboard(year, driver_abbr):
    """
    Shows a dashboard for a single driver's performance over a season.
    """
    try:
        # Call our new analysis function
        chart_pos_html, chart_lap_html = create_driver_improvement_chart(year, driver_abbr)
    
    except Exception as e:
        flash(f"Error processing data for {driver_abbr} in {year}. (Error: {e})", 'danger')
        return redirect(url_for('home'))

    # Render a NEW template, driver.html
    return render_template('driver.html',
                           year=year,
                           driver_abbr=driver_abbr,
                           chart_pos_html=chart_pos_html,
                           chart_lap_html=chart_lap_html
                           )
# --- END NEW ROUTE ---


@app.route('/live')
def live_stats():
    try:
        # Example: Get the latest available race session
        current_year = datetime.datetime.now().year
        session = ff1.get_session(current_year, 'Last', 'R')  # Last race of the year
        session.load(laps=True, telemetry=False, weather=True)

        # Extract key live data
        latest_lap = session.laps.pick_fastest()
        leader = latest_lap['Driver']
        fastest_time = latest_lap['LapTime']
        track = session.event['EventName']
        weather = session.weather_data[-1]

        data = {
            "track": track,
            "leader": leader,
            "fastest_lap": str(fastest_time),
            "temperature": weather['AirTemp'],
            "humidity": weather['Humidity'],
            "rainfall": weather['Rainfall']
        }

        return render_template('live.html', data=data)

    except Exception as e:
        # Handle case when data is not available
        error_msg = f"Live data currently unavailable. Please try again later. ({str(e)})"
        return render_template('live.html', error=error_msg)
    

# -----------
# ML
# -----------
@app.route('/predictor')
def predictor():
    """Renders the prediction input form."""
    return render_template('predictor.html')



# In app.py

@app.route('/run_prediction', methods=['POST'])
def run_prediction():
    """Handles the form, calls Groq, and shows the result."""
    
    try:
        # 1. Get all the data from the form (unchanged)
        driver = request.form['driver']
        start_pos = request.form['starting_position']
        track_name = request.form['track_name']
        conditions = request.form['conditions']
        laps = request.form['laps']
        tyre = request.form['tyre']

        # 2. Craft the prompt for Groq (unchanged)
        system_prompt = "You are an expert Formula 1 analyst for the 'FormulaFever' project."
        
        user_prompt = f"""
        You are an expert Formula 1 analyst for the 'FormulaFever' project. 
        A user is asking for a prediction for an upcoming race.

        Based *only* on the data provided, provide a concise summary and an estimated win probability.
        
        **Input Data:**
        - **Driver:** {driver}
        - **Starting Position:** {start_pos}
        - **Track Name:** {track_name}
        - **Conditions:** {conditions}
        - **Number of Laps:** {laps}
        - **Starting Tyre:** {tyre}

        Please analyze these factors (e.g., how starting position matters at this track,
        tyre strategy, driver's strength in these conditions) and provide your
        analysis in the following exact format:
        
        **AI Summary:** [Your 2-3 sentence expert summary here.]
        **Win Probability:** [Your estimated probability, e.g., "~15%"]
        """

        # 3. Call the Groq API (Using the new, correct model)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model="llama-3.1-8b-instant", # The new, active model
        )
        raw_text = chat_completion.choices[0].message.content

        # 4. Parse the response (This part is already fixed)
        summary = "Could not parse AI summary. (Model output: " + raw_text + ")" 
        probability = "Could not parse AI probability."

        if "**AI Summary:**" in raw_text and "**Win Probability:**" in raw_text: 
            summary_split = raw_text.split("**AI Summary:**")[1]
            summary = summary_split.split("**Win Probability:**")[0].strip()
            probability = raw_text.split("**Win Probability:**")[1].strip()
        else:
            print(f"DEBUG: Parsing failed. Raw text was: {raw_text}")


        # 5. Render the new results page (THIS IS THE UPDATED PART)
        # We are now passing all the original inputs to the template
        return render_template('prediction_result.html',
                               summary=summary,
                               probability=probability,
                               driver=driver,
                               track=track_name,
                               start_pos=start_pos,
                               conditions=conditions,
                               laps=laps,
                               tyre=tyre)

    except Exception as e:
        flash(f"An error occurred while generating the AI prediction: {e}", "danger")
        return redirect(url_for('predictor'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)