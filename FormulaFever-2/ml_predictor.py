import os
import pandas as pd
import fastf1
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load

def load_training_data():
    """
    Loads all available historical race lap data for training the ML model.
    Automatically skips years or races that cannot be fetched.
    """
    os.makedirs('ff1_cache', exist_ok=True)
    fastf1.Cache.enable_cache('ff1_cache')

    all_data = []
    years_to_try = list(range(2018, 2025))  # Fetch 2018–2024 automatically

    for year in years_to_try:
        print(f"\n🟡 Attempting to fetch schedule for {year}...")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            print(f"✅ Found {len(schedule)} races for {year}.")
        except Exception as e:
            print(f"❌ Could not fetch schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            event_name = event['EventName']
            try:
                print(f"   ⏳ Loading {year} {event_name} ...")
                session = fastf1.get_session(year, event_name, 'R')
                session.load(laps=True, telemetry=False, weather=False)
                laps = session.laps
                if laps is None or laps.empty:
                    print(f"   ⚠️ No lap data for {event_name}, skipping.")
                    continue

                laps['Year'] = year
                laps['EventName'] = event_name
                all_data.append(laps)
                print(f"   ✅ Loaded {len(laps)} laps for {event_name}.")
            except Exception as e:
                print(f"   ❌ Skipping {event_name} ({year}): {e}")
                continue

    if not all_data:
        raise ValueError("No data could be loaded. Check FastF1 cache or internet connection.")

    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Successfully loaded {len(df)} total laps from {len(years_to_try)} years.")
    return df


# 
    """Train ML models for lap time and winner prediction."""
    data = load_training_data()

    # --- Clean and prepare data ---
    data = data.dropna(subset=['LapTime'])  # Drop rows with no lap time
    data = data.copy()

    # Label encode categorical columns
    enc_driver = LabelEncoder()
    enc_tyre = LabelEncoder()

    data['Driver_enc'] = enc_driver.fit_transform(data['Driver'])
    data['Tyre_enc'] = enc_tyre.fit_transform(data['Compound'])
    data['LapTime(s)'] = data['LapTime'].dt.total_seconds()

    # Drop any rows that still have NaN values
    data = data.dropna()

    # Select only available numeric columns
    available_cols = [col for col in ['LapNumber', 'Driver_enc', 'Tyre_enc', 'AirTemp', 'TrackTemp'] if col in data.columns]
    X_time = data[available_cols].fillna(0)
    y_time = data['LapTime(s)']

    # --- Train lap time prediction model ---
    X_train, X_test, y_train, y_test = train_test_split(X_time, y_time, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=150, random_state=42)
    reg.fit(X_train, y_train)
    r2_score = reg.score(X_test, y_test)
    print(f"✅ Lap Time Model Trained (R² = {r2_score:.3f})")

    # --- Train winner classification model ---
    avg_laps = data.groupby('Driver')['LapTime(s)'].mean().reset_index()
    avg_laps['Winner'] = (avg_laps['LapTime(s)'] == avg_laps['LapTime(s)'].min()).astype(int)

    X_class = data.groupby('Driver')[['Driver_enc', 'Tyre_enc', 'AirTemp', 'TrackTemp']].mean().fillna(0)
    y_class = avg_laps['Winner']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_class, y_class)
    print("✅ Winner Model Trained Successfully")

    # --- Save models ---
    os.makedirs('models', exist_ok=True)
    dump(reg, 'models/lap_time_model.pkl')
    dump(clf, 'models/winner_model.pkl')
    dump(enc_driver, 'models/enc_driver.pkl')
    dump(enc_tyre, 'models/enc_tyre.pkl')
    print("💾 Models Saved in /models folder")

    return reg, clf, enc_driver, enc_tyre, r2_score
# def train_models():
    """Train ML models for lap time and winner prediction."""
    data = load_training_data()

    # Drop rows without lap time
    data = data.dropna(subset=['LapTime']).copy()

    # Ensure LapTime in seconds exists
    data['LapTime(s)'] = data['LapTime'].dt.total_seconds()

    # Drop rows with NaN lap times
    data = data.dropna(subset=['LapTime(s)'])

    # Encode driver & tyre
    enc_driver = LabelEncoder()
    enc_tyre = LabelEncoder()
    data['Driver_enc'] = enc_driver.fit_transform(data['Driver'])
    data['Tyre_enc'] = enc_tyre.fit_transform(data['Compound'])

    # Check available columns dynamically
    possible_cols = ['LapNumber', 'Driver_enc', 'Tyre_enc', 'AirTemp', 'TrackTemp']
    available_cols = [c for c in possible_cols if c in data.columns]

    if len(data) == 0 or len(available_cols) == 0:
        raise ValueError("❌ No usable data available for training. Check your FastF1 cache or internet.")

    # Prepare training data
    X_time = data[available_cols].fillna(0)
    y_time = data['LapTime(s)']

    # If dataset is too small, skip split
    if len(X_time) < 5:
        print("⚠️ Dataset too small for train/test split. Training on all available data.")
        X_train, X_test, y_train, y_test = X_time, X_time, y_time, y_time
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_time, y_time, test_size=0.2, random_state=42)

    # Train lap time regressor
    reg = RandomForestRegressor(n_estimators=150, random_state=42)
    reg.fit(X_train, y_train)
    r2_score = reg.score(X_test, y_test) if len(X_test) > 0 else 0.0
    print(f"✅ Lap Time Model trained successfully (R² = {r2_score:.3f})")

    # Winner model — classify best average lap
    avg_laps = data.groupby('Driver')['LapTime(s)'].mean().reset_index()
    avg_laps['Winner'] = (avg_laps['LapTime(s)'] == avg_laps['LapTime(s)'].min()).astype(int)

    X_class = data.groupby('Driver')[['Driver_enc', 'Tyre_enc', 'AirTemp', 'TrackTemp']].mean().fillna(0)
    y_class = avg_laps['Winner']

    if len(X_class) == 0 or len(y_class) == 0:
        print("⚠️ Not enough data for winner model. Skipping this part.")
        clf = None
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_class, y_class)
        print("✅ Winner Model trained successfully.")

    # Save models
    os.makedirs('models', exist_ok=True)
    dump(reg, 'models/lap_time_model.pkl')
    if clf is not None:
        dump(clf, 'models/winner_model.pkl')
    dump(enc_driver, 'models/enc_driver.pkl')
    dump(enc_tyre, 'models/enc_tyre.pkl')

    print("💾 Models saved successfully in /models folder.")
    return reg, clf, enc_driver, enc_tyre, r2_score
def train_models():
    """Train ML models for lap time and winner prediction."""
    data = load_training_data()

    # Drop rows without lap time
    data = data.dropna(subset=['LapTime']).copy()
    data['LapTime(s)'] = data['LapTime'].dt.total_seconds()
    data = data.dropna(subset=['LapTime(s)'])

    # Encode driver & tyre
    enc_driver = LabelEncoder()
    enc_tyre = LabelEncoder()
    data['Driver_enc'] = enc_driver.fit_transform(data['Driver'])
    data['Tyre_enc'] = enc_tyre.fit_transform(data['Compound'])

    # Pick only columns that exist
    possible_cols = ['LapNumber', 'Driver_enc', 'Tyre_enc', 'AirTemp', 'TrackTemp']
    available_cols = [c for c in possible_cols if c in data.columns]

    if len(available_cols) == 0:
        raise ValueError("❌ No usable columns found for training (missing telemetry/weather data).")

    # Prepare regression data
    X_time = data[available_cols].fillna(0)
    y_time = data['LapTime(s)']

    # Handle small datasets
    if len(X_time) < 5:
        print("⚠️ Not enough samples for train/test split, training on full data.")
        X_train, X_test, y_train, y_test = X_time, X_time, y_time, y_time
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_time, y_time, test_size=0.2, random_state=42)

    # Train regression model
    reg = RandomForestRegressor(n_estimators=150, random_state=42)
    reg.fit(X_train, y_train)
    r2_score = reg.score(X_test, y_test) if len(X_test) > 0 else 0.0
    print(f"✅ Lap Time Model trained successfully (R² = {r2_score:.3f})")

    # Prepare features dynamically for winner model
    winner_cols = ['Driver_enc', 'Tyre_enc'] + [col for col in ['AirTemp', 'TrackTemp'] if col in data.columns]
    X_class = data.groupby('Driver')[winner_cols].mean().fillna(0)

    avg_laps = data.groupby('Driver')['LapTime(s)'].mean().reset_index()
    avg_laps['Winner'] = (avg_laps['LapTime(s)'] == avg_laps['LapTime(s)'].min()).astype(int)
    y_class = avg_laps['Winner']

    if len(X_class) == 0 or len(y_class) == 0:
        print("⚠️ Not enough data for winner model, skipping classification.")
        clf = None
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_class, y_class)
        print("✅ Winner Model trained successfully.")

    # Save models
    os.makedirs('models', exist_ok=True)
    dump(reg, 'models/lap_time_model.pkl')
    if clf is not None:
        dump(clf, 'models/winner_model.pkl')
    dump(enc_driver, 'models/enc_driver.pkl')
    dump(enc_tyre, 'models/enc_tyre.pkl')

    print("💾 Models saved successfully in /models folder.")
    return reg, clf, enc_driver, enc_tyre, r2_score


# def generate_prediction_summary(driver, tyre, lap, air_temp, track_temp):
    """Generate AI-based summary for a given driver setup."""
    reg = load('models/lap_time_model.pkl')
    clf = load('models/winner_model.pkl')
    enc_driver = load('models/enc_driver.pkl')
    enc_tyre = load('models/enc_tyre.pkl')

    # Get known driver and tyre lists
    known_drivers = list(enc_driver.classes_)
    known_tyres = list(enc_tyre.classes_)

    # Handle unknown driver/tyre gracefully
    if driver not in known_drivers:
        print(f"⚠️ Unknown driver: {driver}. Defaulting to {known_drivers[0]}")
        driver = known_drivers[0]
    if tyre not in known_tyres:
        print(f"⚠️ Unknown tyre: {tyre}. Defaulting to {known_tyres[0]}")
        tyre = known_tyres[0]

    # Encode safely
    driver_encoded = enc_driver.transform([driver])[0]
    tyre_encoded = enc_tyre.transform([tyre])[0]

    # Build feature row
    X_new = pd.DataFrame([[lap, driver_encoded, tyre_encoded, air_temp, track_temp]],
                         columns=['LapNumber', 'Driver_enc', 'Tyre_enc', 'AirTemp', 'TrackTemp'])
    
    predicted_time = reg.predict(X_new)[0]
    win_prob = clf.predict_proba([[driver_encoded, tyre_encoded, air_temp, track_temp]])[0][1] * 100

    summary = {
        "driver": driver,
        "tyre": tyre,
        "lap_number": lap,
        "predicted_lap_time": f"{predicted_time:.3f} sec",
        "winning_probability": f"{win_prob:.2f}%",
        "recommendation": (
            "Excellent consistency — strong chance of podium!" if win_prob > 70
            else "Good performance expected, mid-grid likely."
        ),
        "context": f"Based on weather: Air {air_temp}°C, Track {track_temp}°C"
    }

    return summary
# def generate_prediction_summary(driver, tyre, lap, air_temp, track_temp):
    """Generate AI-based summary for a given driver setup."""
    reg = load('models/lap_time_model.pkl')
    clf = load('models/winner_model.pkl')
    enc_driver = load('models/enc_driver.pkl')
    enc_tyre = load('models/enc_tyre.pkl')

    # Get known driver and tyre lists
    known_drivers = list(enc_driver.classes_)
    known_tyres = list(enc_tyre.classes_)

    # Handle unknown driver/tyre gracefully
    if driver not in known_drivers:
        print(f"⚠️ Unknown driver: {driver}. Defaulting to {known_drivers[0]}")
        driver = known_drivers[0]
    if tyre not in known_tyres:
        print(f"⚠️ Unknown tyre: {tyre}. Defaulting to {known_tyres[0]}")
        tyre = known_tyres[0]

    driver_encoded = enc_driver.transform([driver])[0]
    tyre_encoded = enc_tyre.transform([tyre])[0]

    # --- Match training columns dynamically ---
    expected_features = reg.feature_names_in_.tolist()  # Columns used during training
    print(f"Expected columns: {expected_features}")  # Debug line (optional)

    # Build dataframe with all possible columns
    all_features = {
        'LapNumber': lap,
        'Driver_enc': driver_encoded,
        'Tyre_enc': tyre_encoded,
        'AirTemp': air_temp,
        'TrackTemp': track_temp
    }

    # Keep only features that model expects
    X_new = pd.DataFrame([[all_features[f] for f in expected_features]], columns=expected_features)

    predicted_time = reg.predict(X_new)[0]

    # For classifier, match features dynamically too
    if clf is not None:
        expected_cls_features = clf.feature_names_in_.tolist()
        X_cls = pd.DataFrame([[all_features.get(f, 0) for f in expected_cls_features]], columns=expected_cls_features)
        win_prob = clf.predict_proba(X_cls)[0][1] * 100
    else:
        win_prob = 0

    summary = {
        "driver": driver,
        "tyre": tyre,
        "lap_number": lap,
        "predicted_lap_time": f"{predicted_time:.3f} sec",
        "winning_probability": f"{win_prob:.2f}%",
        "recommendation": (
            "Excellent consistency — strong chance of podium!" if win_prob > 70
            else "Good performance expected, mid-grid likely."
        ),
        "context": f"Based on weather: Air {air_temp}°C, Track {track_temp}°C"
    }

    return summary

import numpy as np
import pandas as pd
from joblib import load
from scipy.special import softmax  # install with: pip install scipy
def generate_prediction_summary(driver, tyre, lap, air_temp, track_temp):
    """Generate AI-based race summary with realistic win probability."""
    # Load models and encoders
    reg = load('models/lap_time_model.pkl')
    enc_driver = load('models/enc_driver.pkl')
    enc_tyre = load('models/enc_tyre.pkl')

    known_drivers = list(enc_driver.classes_)
    known_tyres = list(enc_tyre.classes_)

    if driver not in known_drivers:
        print(f"⚠️ Unknown driver '{driver}', defaulting to {known_drivers[0]}")
        driver = known_drivers[0]
    if tyre not in known_tyres:
        print(f"⚠️ Unknown tyre '{tyre}', defaulting to {known_tyres[0]}")
        tyre = known_tyres[0]

    driver_encoded = enc_driver.transform([driver])[0]
    tyre_encoded = enc_tyre.transform([tyre])[0]

    # Prepare full feature dictionary (safe default)
    feature_dict = {
        'LapNumber': lap,
        'Driver_enc': driver_encoded,
        'Tyre_enc': tyre_encoded,
        'AirTemp': air_temp,
        'TrackTemp': track_temp
    }

    # ---- DYNAMIC COLUMN MATCH ----
    expected_features = reg.feature_names_in_
    X_values = [feature_dict.get(f, 0) for f in expected_features]
    X_new = pd.DataFrame([X_values], columns=expected_features)

    # Predict lap time
    predicted_time = reg.predict(X_new)[0]

    # Simulate probabilities based on all drivers
    all_driver_probs = []
    for d in known_drivers:
        d_enc = enc_driver.transform([d])[0]
        temp_features = feature_dict.copy()
        temp_features['Driver_enc'] = d_enc

        X_temp = pd.DataFrame([[temp_features.get(f, 0) for f in expected_features]], columns=expected_features)
        time_pred = reg.predict(X_temp)[0]
        all_driver_probs.append((d, time_pred))

    df = pd.DataFrame(all_driver_probs, columns=["Driver", "PredictedLapTime"])
    df["WinProb"] = softmax(-df["PredictedLapTime"].values) * 100
    win_prob = float(df.loc[df["Driver"] == driver, "WinProb"].iloc[0])

    # Recommendation logic
    recommendation = (
        "🔥 Excellent consistency — strong podium chance!" if win_prob > 60
        else "⚙️ Good performance expected, mid-grid likely." if win_prob > 25
        else "🚧 Tough conditions — may struggle to finish high."
    )

    summary = {
        "driver": driver,
        "tyre": tyre,
        "lap_number": lap,
        "predicted_lap_time": f"{predicted_time:.3f} sec",
        "winning_probability": f"{win_prob:.2f}%",
        "recommendation": recommendation,
        "context": f"Based on weather: Air {air_temp}°C, Track {track_temp}°C"
    }

    return summary