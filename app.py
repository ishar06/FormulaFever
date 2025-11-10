import os
import datetime
import requests
import fastf1 as ff1
import markdown  # <-- Make sure this is imported
from groq import Groq
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
# --- REMOVED CHART IMPORT ---
from analysis import create_laptime_chart, create_comparison_plots
from flask_caching import Cache
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------
# App setup
# -----------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_change_this'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///formula_community.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["CACHE_TYPE"] = "SimpleCache"
cache = Cache(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# -----------------------------------------------------
# GROQ AI Setup
# -----------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("CRITICAL: GROQ_API_KEY environment variable not set. Please set the key and restart the app.")

try:
    client = Groq(api_key=api_key)
    print("✅ Groq AI Client initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Could not initialize Groq AI Client. Error: {e}")
    raise e

# -----------------------------------------------------
# FASTF1 Setup
# -----------------------------------------------------
try:
    ff1.Cache.enable_cache('ff1_cache/')
    print("FastF1 cache enabled.")
except Exception as e:
    print(f"CRITICAL: Could not enable FastF1 cache: {e}")

# -----------------------------------------------------
# MODELS (Community System)
# -----------------------------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    posts = db.relationship('Post', backref='author', lazy=True)


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    reactions = db.relationship('Reaction', backref='post', lazy=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Reaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    emoji = db.Column(db.String(10), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)

# -----------------------------------------------------
# AI HELPER FUNCTION (MODIFIED)
# -----------------------------------------------------
def get_ai_driver_analysis(year, driver_abbr, analysis_focus):
    """
    Calls Groq to get a qualitative summary of a driver's season.
    This prompt is now simpler and more direct.
    """
    try:
        # --- NEW, SIMPLIFIED PROMPT ---
        system_prompt = """
You are a factual Formula 1 analyst and database expert for the 'FormulaFever' project.
Your goal is to provide an accurate, high-quality analysis of a driver's season based **on your internal knowledge**.
You MUST be accurate. Do NOT invent statistics. If you do not know a specific stat, you MUST write "N/A".
"""

        # Build a dynamic prompt
        user_prompt = f"""
        **Task:** Generate a performance analysis for a Formula 1 driver.

        **Inputs:**
        - **Driver:** {driver_abbr}
        - **Season:** {year}
        - **Analysis Focus:** {analysis_focus}

        **Instructions:**
        1.  **Use Your Knowledge:** Use your internal F1 knowledge base to find *factual* statistics for `{driver_abbr}` during the `{year}` season.
        2.  **Be Accurate:** Provide *only* real statistics. If you cannot find a specific stat, write "N/A".
        3.  **Stay Focused:** The *entire* analysis must be about `{driver_abbr}`. Do not mention other drivers (like Leclerc, Hamilton, etc.) unless it is essential for context.

        **Required Output Format (Strict Markdown):**
        
        ## 🏎️ {driver_abbr}'s {year} Season Analysis
        
        (A 2-3 sentence summary paragraph of their `{year}` season, focusing *only* on `{driver_abbr}`.)
        
        ### 📊 Key Season Statistics
        
        (A markdown table with the following *factual* stats for `{driver_abbr}` in `{year}`. You MUST find this information from your knowledge base.)
        | Stat | Value |
        | :--- | :--- |
        | Team | (Team `{driver_abbr}` drove for in `{year}`) |
        | Championship Position | (Final championship position) |
        | Wins | (Total wins) |
        | Podiums | (Total podiums) |
        | Poles | (Total poles) |
        | DNFs | (Total DNFs) |
        
        ### ⭐ Strengths (based on `{analysis_focus}`)
        
        * (Bullet point 1 focusing *only* on `{driver_abbr}`'s strengths)
        * (Bullet point 2 focusing *only* on `{driver_abbr}`'s strengths)
        
        ### ⚠️ Weaknesses (based on `{analysis_focus}`)
        
        * (Bullet point 1 focusing *only* on `{driver_abbr}`'s weaknesses)
        * (Bullet point 2 focusing *only* on `{driver_abbr}`'s weaknesses)
        
        ### 🏁 Season Conclusion
        
        (A final summary paragraph about `{driver_abbr}`'s `{year}` season.)
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            # --- FIX 1: FORCE DETERMINISTIC, FACTUAL OUTPUT ---
            temperature=0
        )
        return chat_completion.choices[0].message.content, None
    except Exception as e:
        print(f"--- ERROR: Groq AI analysis failed: {e} ---")
        return None, str(e)


# -----------------------------------------------------
# MAIN ROUTES (Your FormulaFever logic)
# -----------------------------------------------------

@app.route('/')
def home():
    # Pass current year to the form
    return render_template('home.html', current_year=datetime.datetime.now().year)


@app.route('/get_standings')
@cache.cached(timeout=3600)
def get_standings():
    """
    (This route is correct and unchanged)
    Called by JavaScript to load standings.
    Handles API timeouts gracefully.
    """
    # --- THIS IS THE FIX for Problem 1 ---
    url = "https://api.jolpi.ca/ergast/f1/current/driverStandings.json"
    # -----------------------------------
    
    standings = None
    season = None
    error = None

    try:
        print("--- FETCHING LIVE STANDINGS (NOT FROM CACHE) ---")
        response = requests.get(url, timeout=45)
        response.raise_for_status()

        if not response.text:
            raise ValueError("API returned an empty response.")

        data = response.json()
        standings_list = data.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [])

        if standings_list:
            standings = standings_list[0]['DriverStandings']
            season = standings_list[0]['season']
        else:
            error = "Could not find any standings data for the current season."

    except requests.exceptions.Timeout:
        print("--- WARNING: Standings API call timed out. ---")
        error = "The request for live standings timed out. The API might be down. Please try again in a moment."

    except requests.exceptions.RequestException as e:
        print(f"--- WARNING: Could not load driver standings. RequestException: {e} ---")
        error = f"Could not retrieve standings. An API error occurred: {e}"

    except Exception as e:
        print(f"--- WARNING: Could not load driver standings. General Exception: {e} ---")
        error = f"An unexpected error occurred while processing standings data: {e}"

    # This renders the partial HTML file with the error or the table
    return render_template('_standings_partial.html', standings=standings, season=season, error=error)


# ... (routes /race_search, /race, /comparator, /compare are unchanged) ...
@app.route('/race_search')
def race_search():
    year = request.args.get('year')
    race_name_input = request.args.get('race_name')

    if not year or not race_name_input:
        flash('Both Year and Race Name are required.', 'danger')
        return redirect(url_for('home'))

    try:
        year_int = int(year)
        schedule = ff1.get_event_schedule(year_int, include_testing=False)
        valid_names = set(schedule['Location'].str.lower()) | set(schedule['Country'].str.lower()) | set(schedule['EventName'].str.lower())
        if race_name_input.lower() not in valid_names:
            flash(f"'{race_name_input}' is not a valid race for {year_int}.", 'danger')
            return redirect(url_for('home'))
        return redirect(url_for('race_dashboard', year=year_int, race_name=race_name_input.capitalize()))
    except Exception as e:
        flash(f"Error validating race: {e}", 'danger')
        return redirect(url_for('home'))


@app.route('/race/<int:year>/<string:race_name>')
def race_dashboard(year, race_name):
    try:
        lap_chart_html, summary = create_laptime_chart(year, race_name)
        return render_template('race.html', year=year, race_name=race_name, lap_chart_html=lap_chart_html, summary=summary)
    except Exception as e:
        flash(f"Error: {e}", 'danger')
        return redirect(url_for('home'))


@app.route('/comparator')
def comparator():
    return render_template('comparator.html')


@app.route('/compare', methods=['POST'])
def compare_laps():
    try:
        year = int(request.form['year'])
        gp = request.form['gp']
        session = request.form['session']
        driver1 = request.form['driver1'].upper()
        driver2 = request.form['driver2'].upper()
        plot_data = create_comparison_plots(year, gp, session, driver1, driver2)
        if "error" in plot_data:
            return render_template('comparator.html', error_message=plot_data['error'])
        return render_template('comparator.html', delta_plot=plot_data['delta_plot'], telemetry_plot=plot_data['telemetry_plot'])
    except Exception as e:
        return render_template('comparator.html', error_message=str(e))


# === DRIVER ANALYSIS FORM ROUTE (UNCHANGED) ===
@app.route('/driver_analysis')
def driver_analysis_form():
    """Renders the new, dedicated driver analysis form page."""
    # Pass datetime to the template so 'max' year in form is dynamic
    return render_template('driver_analysis_form.html', datetime=datetime)


# === (MODIFIED) DRIVER ANALYSIS RESULTS ROUTE ===
# --- FIX 2: ADD CACHING TO THE ROUTE ---
@app.route('/driver_analysis/result')
@cache.cached(timeout=3600)  # Cache this result for 1 hour
def driver_analysis_result():
    """
    (FAST) Shows the AI summary IMMEDIATELY.
    The charts have been removed.
    """
    # Get all parameters from the form
    year = request.args.get('year', type=int)
    driver_abbr = request.args.get('driver_abbr')
    # --- REMOVED COMPARISON DRIVER ---
    analysis_focus = request.args.get('analysis_focus', default='Overall Performance')

    # Basic validation
    if not year or not driver_abbr:
        flash('Year and Driver Abbreviation are required.', 'danger')
        return redirect(url_for('driver_analysis_form'))

    driver_abbr = driver_abbr.upper()

    ai_summary_html = None
    ai_error = None

    try:
        # 1. Get the AI summary (the fast part)
        print(f"--- RUNNING Groq AI analysis for {driver_abbr} {year} (Not from cache) ---")
        # --- UPDATED FUNCTION CALL ---
        ai_summary_md, ai_error = get_ai_driver_analysis(year, driver_abbr, analysis_focus)
        
        # 2. Convert markdown summary to HTML
        if ai_summary_md:
            # Added 'tables' to the extensions to make sure our new table works
            ai_summary_html = markdown.markdown(ai_summary_md, extensions=['fenced_code', 'tables'])

    except Exception as e:
        print(f"--- ERROR: Groq AI analysis failed: {e} ---")
        ai_error = str(e)

    # 3. Render the results page
    # All chart-related variables have been removed
    return render_template(
        'driver.html',
        year=year,
        driver_abbr=driver_abbr,
        analysis_focus=analysis_focus,
        ai_summary_html=ai_summary_html,
        ai_error=ai_error
    )


# === (REMOVED) /get_driver_charts ROUTE ===
# The slow chart-generating route is now gone.


# ... (rest of your routes: /live, /predictor, /run_prediction, /about, etc.) ...
@app.route('/live')
def live_stats():
    try:
        current_year = datetime.datetime.now().year
        session = ff1.get_session(current_year, 'Last', 'R')
        session.load(laps=True, telemetry=False, weather=True)
        latest_lap = session.laps.pick_fastest()
        weather = session.weather_data[-1]
        data = {
            "track": session.event['EventName'],
            "leader": latest_lap['Driver'],
            "fastest_lap": str(latest_lap['LapTime']),
            "temperature": weather['AirTemp'],
            "humidity": weather['Humidity'],
            "rainfall": weather['Rainfall']
        }
        return render_template('live.html', data=data)
    except Exception as e:
        return render_template('live.html', error=f"Live data unavailable ({e})")


# -----------
# ML
# -----------
@app.route('/predictor')
def predictor():
    """Renders the prediction input form."""
    return render_template('predictor.html')


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
            model="llama-3.1-8b-instant",  # The new, active model
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


# -----------------------------------------------------
# COMMUNITY ROUTES
# -----------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form.get("confirm_password")  # Get the confirmation field

        # --- START VALIDATION ---

        # 1. Check for existing user (prevents unique constraint crash)
        if User.query.filter_by(email=email).first():
            flash("That email address is already in use.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("That username is already taken.", "danger")
            return redirect(url_for("register"))

        # 2. Check passwords
        if not confirm_password:
            flash("Please confirm your password.", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match. Please try again.", "danger")
            return redirect(url_for("register"))

        if len(password) < 8:  # Simple length check
            flash("Password must be at least 8 characters long.", "danger")
            return redirect(url_for("register"))

        # --- END VALIDATION ---

        # All checks passed! Hash the password and create the user.
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        user = User(username=username, email=email, password=hashed_password)

        db.session.add(user)
        db.session.commit()

        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("community"))
        flash("Invalid credentials. Try again.", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route("/community", methods=["GET", "POST"])
@login_required
def community():
    if request.method == "POST":
        content = request.form["content"]

        # AI check (Your existing logic is fine)
        try:
            check_prompt = f"Is this message about Formula 1 or motorsports? Reply only 'yes' or 'no':\n\n{content}"
            ai_response = client.chat.completions.create(
                messages=[{"role": "user", "content": check_prompt}],
                model="llama-3.1-8b-instant"
            )
            verdict = ai_response.choices[0].message.content.strip().lower()

            if verdict != "yes":
                flash("❌ Please keep discussions related to Formula 1 only.", "danger")
                return redirect(url_for("community"))
        except Exception as e:
            flash(f"AI moderation failed: {e}", "warning")

        # Save post (Your existing logic is fine)
        post = Post(content=content, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash("Your message has been posted!", "success")
        return redirect(url_for("community"))

    # --- START OF FIX (GET Request) ---

    posts = Post.query.order_by(Post.timestamp.desc()).all()

    # Create a dictionary to hold the reaction counts for each post
    reaction_counts = {}
    for post in posts:
        # Query the counts for this specific post
        counts = (
            db.session.query(Reaction.emoji, db.func.count(Reaction.id))
            .filter_by(post_id=post.id)
            .group_by(Reaction.emoji)
            .all()
        )
        # Format the counts into the string your JS expects
        # e.g., [('👍', 5), ('❤️', 2)] -> "👍 5  ❤️ 2"
        count_string = '  '.join([f"{emoji} {count}" for emoji, count in counts])
        reaction_counts[post.id] = count_string

    # Pass the new 'reaction_counts' dictionary to the template
    return render_template("community.html", posts=posts, reaction_counts=reaction_counts)

    # --- END OF FIX ---


@app.route('/react', methods=['POST'])
@login_required
def react():
    try:
        data = request.get_json()
        post_id = data.get('post_id')
        emoji = data.get('emoji')

        if not post_id or not emoji:
            return jsonify({"error": "Invalid data"}), 400

        # Check if user already reacted
        existing = Reaction.query.filter_by(user_id=current_user.id, post_id=post_id).first()

        if existing:
            # If they click the same emoji, remove the reaction (toggle off)
            if existing.emoji == emoji:
                db.session.delete(existing)
            else:
                existing.emoji = emoji  # Update to the new emoji
        else:
            # No existing reaction, so add a new one
            new_reaction = Reaction(emoji=emoji, user_id=current_user.id, post_id=post_id)
            db.session.add(new_reaction)

        db.session.commit()

        # Recalculate the counts for this post
        counts = (
            db.session.query(Reaction.emoji, db.func.count(Reaction.id))
            .filter_by(post_id=post_id)
            .group_by(Reaction.emoji)
            .all()
        )

        # Return the new counts. This will be an empty dict {} if no reactions are left.
        return jsonify({emoji: count for emoji, count in counts})

    except Exception as e:
        db.session.rollback()  # Important: undo any failed changes
        print(f"*** SERVER ERROR IN /react: {e} ***")  # Log the error
        return jsonify({"error": "Server error"}), 500


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)