import os
import datetime
import requests
import fastf1 as ff1
from groq import Groq
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from analysis import create_laptime_chart, create_comparison_plots, create_driver_improvement_chart

# -----------------------------------------------------
# App setup
# -----------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_change_this'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///formula_community.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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
# MAIN ROUTES (Your FormulaFever logic)
# -----------------------------------------------------
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
    except Exception as e:
        return render_template('home.html', standings=None, error=f"Error loading data: {e}")

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

@app.route('/driver_search')
def driver_search():
    year = request.args.get('year')
    driver_abbr = request.args.get('driver_abbr')
    if not year or not driver_abbr:
        flash('Year and Driver Abbreviation are required.', 'danger')
        return redirect(url_for('home'))
    try:
        year_int = int(year)
        driver_abbr_upper = driver_abbr.upper()
        return redirect(url_for('driver_dashboard', year=year_int, driver_abbr=driver_abbr_upper))
    except Exception as e:
        flash(str(e), 'danger')
        return redirect(url_for('home'))

@app.route('/driver/<int:year>/<string:driver_abbr>')
def driver_dashboard(year, driver_abbr):
    try:
        chart_pos_html, chart_lap_html = create_driver_improvement_chart(year, driver_abbr)
        return render_template('driver.html', year=year, driver_abbr=driver_abbr, chart_pos_html=chart_pos_html, chart_lap_html=chart_lap_html)
    except Exception as e:
        flash(f"Error: {e}", 'danger')
        return redirect(url_for('home'))

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

# -----------------------------------------------------
# COMMUNITY ROUTES
# -----------------------------------------------------

# In app.py

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form.get("confirm_password") # Get the confirmation field

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

        if len(password) < 8: # Simple length check
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




# In app.py

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

# In app.py

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
        print(f"*** SERVER ERROR IN /react: {e} ***") # Log the error
        return jsonify({"error": "Server error"}), 500

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True) 