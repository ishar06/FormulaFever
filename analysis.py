# analysis.py
import fastf1 as ff1
import plotly.express as px
import pandas as pd
import fastf1.plotting
import fastf1.utils as utils
from matplotlib.figure import Figure
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from fastf1.core import DataNotLoadedError

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




def create_comparison_plots(year, gp, session, driver1, driver2):
    """
    Fetches F1 data and generates comparison plot images as Base64 strings.
    Returns a dictionary of plot strings.
    """
    
    try:
        # --- 1. Load Session Data ---
        print(f"Loading session: {year} {gp} {session}")
        session_data = ff1.get_session(year, gp, session)
        session_data.load()
        print("Session loaded. Processing data...")

        # --- 2. Get Fastest Laps for Both Drivers ---
        lap_d1 = session_data.laps.pick_drivers(driver1).pick_fastest()
        lap_d2 = session_data.laps.pick_drivers(driver2).pick_fastest()
        
        if lap_d1 is None or lap_d2 is None:
            return {"error": f"Could not find fastest lap for one of the drivers. Check abbreviations."}

        # --- 3. Get Telemetry Data ---
        tel_d1 = lap_d1.get_car_data().add_distance()
        tel_d2 = lap_d2.get_car_data().add_distance()
        
        color_d1 = fastf1.plotting.get_team_color(lap_d1['Team'], session=session_data)
        color_d2 = fastf1.plotting.get_team_color(lap_d2['Team'], session=session_data)

        # --- 4. Generate the Delta Time Plot ---
        print("Generating Delta Plot...")
        delta_time, ref_tel, comp_tel = utils.delta_time(lap_d1, lap_d2)
        
        fig_delta = Figure(figsize=(10, 3))
        ax_delta = fig_delta.subplots()
        
        ax_delta.plot(ref_tel['Distance'], delta_time, color='white')
        ax_delta.axhline(0, color='white', linestyle='--')
        ax_delta.set_ylabel(f"Delta Time ({driver2} vs {driver1})")
        ax_delta.set_xlabel("Distance (m)")
        fig_delta.suptitle(f"Fastest Lap Delta: {driver1} vs {driver2}", color='white')
        
        # Style the delta plot
        fig_delta.patch.set_facecolor('#0d1117') 
        ax_delta.set_facecolor('#0d1117')
        ax_delta.tick_params(axis='x', colors='white')
        ax_delta.tick_params(axis='y', colors='white')
        ax_delta.spines['top'].set_visible(False)
        ax_delta.spines['right'].set_visible(False)
        ax_delta.spines['bottom'].set_color('white')
        ax_delta.spines['left'].set_color('white')
        ax_delta.title.set_color('white')
        ax_delta.xaxis.label.set_color('white')
        ax_delta.yaxis.label.set_color('white')
        fig_delta.tight_layout()

        # --- 5. Generate the Telemetry Plot (Speed, Throttle, Brake) ---
        print("Generating Telemetry Plot...")
        
        fig_tel = Figure(figsize=(10, 8))
        axes_tel = fig_tel.subplots(nrows=3, sharex=True)
        
        # Speed
        axes_tel[0].plot(tel_d1['Distance'], tel_d1['Speed'], label=driver1, color=color_d1)
        axes_tel[0].plot(tel_d2['Distance'], tel_d2['Speed'], label=driver2, color=color_d2)
        axes_tel[0].set_ylabel('Speed (Km/h)')
        axes_tel[0].legend(loc='lower right')
        
        # Throttle
        axes_tel[1].plot(tel_d1['Distance'], tel_d1['Throttle'], label=driver1, color=color_d1)
        axes_tel[1].plot(tel_d2['Distance'], tel_d2['Throttle'], label=driver2, color=color_d2)
        axes_tel[1].set_ylabel('Throttle (%)')
        
        # Brake
        axes_tel[2].plot(tel_d1['Distance'], tel_d1['Brake'], label=driver1, color=color_d1)
        axes_tel[2].plot(tel_d2['Distance'], tel_d2['Brake'], label=driver2, color=color_d2) 
        axes_tel[2].set_ylabel('Brake')
        axes_tel[2].set_xlabel('Distance (m)')

        fig_tel.suptitle(f"Telemetry Comparison: {driver1} vs {driver2}", color='white')
        
        # Style the telemetry plots
        fig_tel.patch.set_facecolor('#0d1117')
        for ax in axes_tel:
            ax.set_facecolor('#0d1117')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            if ax.get_legend():
                legend = ax.get_legend()
                for text in legend.get_texts():
                    text.set_color('white')
        fig_tel.tight_layout()


        # --- 6. Convert Plots to Base64 ---
        print("Converting plots to Base64...")
        
        # === THIS IS THE FIX (The function definition was missing) ===
        # Function to save figure to a Base64 string
        def fig_to_base_64(fig):
            buf = io.BytesIO()
            FigureCanvas(fig).print_png(buf) # Save plot to buffer
            buf.seek(0)
            data = base64.b64encode(buf.getvalue()).decode('ascii')
            return data
        # === END OF FIX ===

        plot_delta_b64 = fig_to_base_64(fig_delta)
        plot_telemetry_b64 = fig_to_base_64(fig_tel)
        
        print("Plots generated successfully.")
        
        return {
            "delta_plot": plot_delta_b64,
            "telemetry_plot": plot_telemetry_b64
        }
    
    except Exception as e:
        print(f"Error in create_comparison_plots: {e}") # For logging
        return {"error": str(e)}
    
def create_driver_improvement_chart(year, driver_abbr):
    """
    Creates charts for a single driver's performance over a season.
    (This is the robust, updated version)
    """
    print(f"Loading season {year} for driver {driver_abbr}...")
    
    # Get the schedule for the year
    schedule = ff1.get_event_schedule(year, include_testing=False)
    
    # Filter for races that have already happened
    completed_races = schedule[schedule['EventDate'] < pd.Timestamp.now()]
    
    if completed_races.empty:
        # This might happen if you run for 2025 and no races are "completed" yet
         raise ValueError(f"No completed races found for {year}.")

    driver_results = []

    # Loop through each completed race
    for _, event in completed_races.iterrows():
        race_name = event['EventName']
        print(f"Processing... {race_name}")
        
        try:
            # Load the session and results
            session = ff1.get_session(year, race_name, 'R')
            session.load(laps=True, telemetry=False, weather=False)
            
            # --- NEW: Check if driver is in results ---
            # session.results['Abbreviation'] lists all drivers who finished
            if driver_abbr not in session.results['Abbreviation'].values:
                print(f"INFO: {driver_abbr} did not participate or finish {race_name}.")
                continue # Skip this race

            # Get this driver's finishing position
            driver_result = session.results.loc[session.results['Abbreviation'] == driver_abbr].iloc[0]
            position = driver_result['Position']
            
            # Get this driver's laps
            driver_laps = session.laps.pick_drivers([driver_abbr]).pick_quicklaps()
            
            avg_lap = None
            if not driver_laps.empty:
                avg_lap = driver_laps['LapTime'].dt.total_seconds().mean()

            driver_results.append({
                'Race': race_name,
                'AvgLapTime(s)': avg_lap,
                'Position': position
            })
            
        except DataNotLoadedError:
            print(f"ERROR: Could not load FastF1 data for {race_name}. Is your internet working?")
        except KeyError:
            print(f"ERROR: KeyError processing {race_name} for {driver_abbr}. Driver data missing?")
        except Exception as e:
            # General catch-all
            print(f"CRITICAL ERROR processing {race_name} for {driver_abbr}: {e}")

    if not driver_results:
        print(f"CRITICAL: driver_results list is empty after loop for {driver_abbr} in {year}.")
        raise ValueError(f"No data found for {driver_abbr} in {year}.")

    # --- Create DataFrame and Charts ---
    df = pd.DataFrame(driver_results)

    # Chart 1: Finishing Position (Bar Chart)
    fig_pos = px.bar(df,
                     x='Race',
                     y='Position',
                     title=f'{driver_abbr} Finishing Positions ({year})',
                     labels={'Position': 'Finishing Position'}
                    )
    fig_pos.update_yaxes(autorange="reversed")
    fig_pos.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # Chart 2: Average Lap Time (Line Chart)
    fig_lap = px.line(df,
                      x='Race',
                      y='AvgLapTime(s)',
                      title=f'{driver_abbr} Average "Quick Lap" Time ({year})',
                      labels={'AvgLapTime(s)': 'Avg. Lap Time (Seconds)'},
                      markers=True
                     )
    fig_lap.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    chart_pos_html = fig_pos.to_html(full_html=False, include_plotlyjs='cdn')
    chart_lap_html = fig_lap.to_html(full_html=False, include_plotlyjs='cdn')

    print(f"Charts created for {driver_abbr}.")
    
    return chart_pos_html, chart_lap_html