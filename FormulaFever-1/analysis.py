# analysis.py
import fastf1 as ff1
import plotly.express as px
import pandas as pd

# The ff1.Cache.enable_cache() line is GONE from here.
# It's now in app.py, where it belongs.

def create_laptime_chart(year, race_name):
    """
    Creates an interactive Plotly chart and a text summary of the race.
    """
    print(f"Loading session: {year} {race_name}")
    
    # We now trust that year and race_name are valid
    session = ff1.get_session(year, race_name, 'R')
    session.load(laps=True, telemetry=False, weather=False) 
    print("Session loaded. Processing data...")

    laps = session.laps
    
    # Check if there is any lap data
    if laps is None or laps.empty:
        raise ValueError("No lap data found for this session.")

    top_5_drivers = session.results[:5]['Abbreviation'].tolist()
    laps_df = laps.pick_drivers(top_5_drivers).pick_quicklaps()
    
    if laps_df.empty:
        raise ValueError("No valid 'quick laps' found for the top 5. (e.g., many DNFs)")

    laps_df['LapTime(s)'] = laps_df['LapTime'].dt.total_seconds()

    # --- Plotly Chart Creation ---
    fig = px.line(laps_df, 
                  x='LapNumber', 
                  y='LapTime(s)', 
                  color='Driver',
                  title=f'{year} {race_name} Lap Time Comparison (Top 5)',
                  labels={'LapNumber': 'Lap Number', 'LapTime(s)': 'Lap Time (Seconds)'}
                 )
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    print("Chart created. Generating summary...")

    # --- GENERATE RACE SUMMARY ---
    avg_laps = laps_df.groupby('Driver')['LapTime(s)'].mean()
    fastest_avg_driver = avg_laps.idxmin()
    fastest_avg_time = avg_laps.min()
    
    consistency = laps_df.groupby('Driver')['LapTime(s)'].std()
    most_consistent_driver = consistency.idxmin()
    most_consistent_std = consistency.min()

    fastest_lap_row = laps_df.loc[laps_df['LapTime(s)'].idxmin()]
    fastest_lap_driver = fastest_lap_row['Driver']
    fastest_lap_time = fastest_lap_row['LapTime(s)']
    fastest_lap_number = fastest_lap_row['LapNumber']

    summary = {
        'fastest_avg_driver': fastest_avg_driver,
        'fastest_avg_time': f"{fastest_avg_time:.3f}s",
        'most_consistent_driver': most_consistent_driver,
        'most_consistent_std': f"+/- {most_consistent_std:.3f}s",
        'fastest_lap_driver': fastest_lap_driver,
        'fastest_lap_time': f"{fastest_lap_time:.3f}s",
        'fastest_lap_number': int(fastest_lap_number)
    }
    
    print("Summary generated successfully.")
    
    return graph_html, summary