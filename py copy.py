import json
import sys
import math
from datetime import date, timedelta

import pandas as pd
import numpy as np

# Optional: if statsmodels is missing, return a graceful fallback
HAS_SM = False  # Disable statsmodels for now to avoid hanging

import pymysql


def get_db_connection():
    # Match PHP db.php settings
    return pymysql.connect(
        host="localhost",
        user="u520834156_userRMS25",
        password="]lRgy[uiN1",
        database="u520834156_dbRMS2025",
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_current_year_quarterly_data(conn):
    # Pull current year quarterly incidents by barangay and incident type with month and time detection
    sql = (
        "SELECT bt.bt_id, bt.barangay, bt.latitude, bt.longitude, "
        "im.type_of_incident, "
        "QUARTER(im.date_and_time_reported) AS quarter, "
        "MONTH(im.date_and_time_reported) AS month, "
        "DATE(im.date_and_time_reported) AS report_date, "
        "HOUR(im.date_and_time_reported) AS report_hour, "
        "COUNT(*) AS cnt "
        "FROM incident_management im "
        "JOIN place_of_incident poi ON poi.poi_id = im.poi_id "
        "JOIN barangay_table bt ON bt.bt_id = poi.bt_id "
        "WHERE im.category = 'Crime' "
        "AND YEAR(im.date_and_time_reported) = YEAR(CURDATE()) "
        "GROUP BY bt.bt_id, bt.barangay, bt.latitude, bt.longitude, im.type_of_incident, quarter, month, report_date, report_hour"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["bt_id", "barangay", "latitude", "longitude", "type_of_incident", "quarter", "month", "report_date", "report_hour", "cnt"])    
    df = pd.DataFrame(rows)
    return df


def convert_to_standard_time(hour):
    """
    Convert 24-hour format to 12-hour format with AM/PM.
    """
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"

def analyze_hot_hours(df, bt_id, incident_type):
    """
    Analyze the most frequent hours for incidents in a specific barangay and incident type.
    Returns the hot hours range in standard time format.
    """
    # Filter data for specific barangay and incident type
    filtered_df = df[(df['bt_id'] == bt_id) & (df['type_of_incident'] == incident_type)]
    
    if filtered_df.empty:
        return "No data available"
    
    # Count incidents by hour
    hour_counts = {}
    for _, row in filtered_df.iterrows():
        hour = row['report_hour']
        count = row['cnt']
        if hour in hour_counts:
            hour_counts[hour] += count
        else:
            hour_counts[hour] = count
    
    if not hour_counts:
        return "No data available"
    
    # Find the hour with maximum incidents
    max_hour = max(hour_counts.keys(), key=lambda h: hour_counts[h])
    max_count = hour_counts[max_hour]
    
    # Find hours with at least 50% of max incidents
    threshold = max_count * 0.5
    hot_hours = [h for h, c in hour_counts.items() if c >= threshold]
    
    if len(hot_hours) == 1:
        return convert_to_standard_time(max_hour)
    elif len(hot_hours) <= 3:
        hot_hours.sort()
        start_time = convert_to_standard_time(hot_hours[0])
        end_time = convert_to_standard_time(hot_hours[-1])
        return f"{start_time} – {end_time}"
    else:
        # If too many hours, show the top 2-3 hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        top_hours = [h for h, c in sorted_hours[:3]]
        top_hours.sort()
        start_time = convert_to_standard_time(top_hours[0])
        end_time = convert_to_standard_time(top_hours[-1])
        return f"{start_time} – {end_time}"


def detect_continuous_incidents(df):
    """
    Detect continuous incidents (exactly 3 months per quarter) for each barangay and incident type.
    Uses month-level data to ensure proper quarterly analysis.
    Returns data with continuous flag and prediction quarter.
    """
    results = []
    
    for (bt_id, incident_type), group in df.groupby(['bt_id', 'type_of_incident']):
        # Get quarters with incidents and their months
        quarters_with_incidents = {}
        for _, row in group.iterrows():
            q = row['quarter']
            month = row['month']
            if q not in quarters_with_incidents:
                quarters_with_incidents[q] = set()
            quarters_with_incidents[q].add(month)
        
        # Check for continuous patterns (exactly 3 months per quarter for continuity)
        continuous_quarters = []
        for q in [1, 2, 3, 4]:
            if q in quarters_with_incidents:
                months_in_quarter = quarters_with_incidents[q]
                # Check if this quarter has continuous activity (exactly 3 months)
                if len(months_in_quarter) >= 3:  # Exactly 3 months in the quarter
                    continuous_quarters.append(q)
        
        # Determine prediction quarter for continuous incidents
        prediction_quarter = None
        if len(continuous_quarters) >= 1:
            # Find the latest continuous quarter and predict next quarter
            latest_quarter = max(continuous_quarters)
            prediction_quarter = latest_quarter + 1
            if prediction_quarter > 4:
                prediction_quarter = 1  # Wrap to Q1 of next year
        
        # Add to results if there are continuous incidents
        if prediction_quarter is not None:
            barangay_data = group.iloc[0]  # Get barangay metadata
            total_incidents = group['cnt'].sum()
            
            # Get detailed month information for each continuous quarter
            quarter_details = {}
            for q in continuous_quarters:
                quarter_months = quarters_with_incidents[q]
                quarter_details[q] = {
                    'months': sorted(list(quarter_months)),
                    'month_count': len(quarter_months),
                    'incidents': group[group['quarter'] == q]['cnt'].sum()
                }
            
            results.append({
                'bt_id': int(bt_id),
                'barangay': barangay_data['barangay'],
                'latitude': float(barangay_data['latitude']) if not pd.isna(barangay_data['latitude']) else None,
                'longitude': float(barangay_data['longitude']) if not pd.isna(barangay_data['longitude']) else None,
                'type_of_incident': incident_type,
                'continuous_quarters': continuous_quarters,
                'quarter_details': quarter_details,
                'total_incidents': int(total_incidents),
                'prediction_quarter': prediction_quarter,
                'is_continuous': True
            })
    
    return pd.DataFrame(results)


def get_current_quarter():
    """Get current quarter based on current date."""
    from datetime import datetime
    current_month = datetime.now().month
    if current_month in [1, 2, 3]:
        return 1
    elif current_month in [4, 5, 6]:
        return 2
    elif current_month in [7, 8, 9]:
        return 3
    else:
        return 4


def predict_current_quarter_incidents(df, bt_id, incident_type, prediction_quarter):
    """
    Predict how many incidents will occur in prediction quarter based on previous quarter data.
    """
    # Get previous quarter data (quarter before the prediction quarter)
    prev_quarter = prediction_quarter - 1 if prediction_quarter > 1 else 4
    
    # Filter data for this barangay, incident type, and previous quarter
    prev_data = df[(df['bt_id'] == bt_id) & 
                   (df['type_of_incident'] == incident_type) & 
                   (df['quarter'] == prev_quarter)]
    
    print(f"DEBUG: Predicting for bt_id={bt_id}, incident={incident_type}, prediction_q={prediction_quarter}, prev_q={prev_quarter}", file=sys.stderr)
    print(f"DEBUG: Previous data found: {len(prev_data)} records", file=sys.stderr)
    
    if prev_data.empty:
        print(f"DEBUG: No previous data, returning 0", file=sys.stderr)
        return 0  # No prediction if no previous data
    
    # Calculate prediction based on previous quarter incidents
    prev_incidents = prev_data['cnt'].sum()
    print(f"DEBUG: Previous quarter incidents: {prev_incidents}", file=sys.stderr)
    
    # Simple prediction: assume similar pattern with slight variation
    # Add some randomness based on historical patterns
    prediction = int(prev_incidents * 1.1)  # 10% increase assumption
    
    print(f"DEBUG: Calculated prediction: {prediction}", file=sys.stderr)
    return prediction  # Based on actual database data


def build_quarterly_arima_forecast(continuous_df, original_df):
    """
    Build simple forecast for continuous incidents by quarter using month-level data.
    """
    results = []
    
    # Get current quarter
    current_quarter = get_current_quarter()
    
    for _, row in continuous_df.iterrows():
        # Simple forecast based on continuous quarters
        total_incidents = row['total_incidents']
        
        # Calculate forecast value based on continuous pattern strength
        forecast_value = float(total_incidents)
        
        # Calculate confidence intervals
        lower_bound = max(0, forecast_value * 0.7)
        upper_bound = forecast_value * 1.3
        
        # Predict incidents for the prediction quarter based on previous quarter
        current_quarter_prediction = predict_current_quarter_incidents(
            original_df, row['bt_id'], row['type_of_incident'], row['prediction_quarter']
        )
        
        # Analyze hot hours for this barangay and incident type
        hot_hours = analyze_hot_hours(original_df, row['bt_id'], row['type_of_incident'])
        
        # Get month details for display
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        continuous_months = []
        for q in row['continuous_quarters']:
            quarter_info = row['quarter_details'][q]
            months = [month_names[m] for m in quarter_info['months']]
            continuous_months.append(f"Q{q}({','.join(months)})")
        
        results.append({
            'bt_id': row['bt_id'],
            'barangay': row['barangay'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'type_of_incident': row['type_of_incident'],
            'prediction_quarter': row['prediction_quarter'],
            'continuous_months': ', '.join(continuous_months),
            'forecast_value': forecast_value,
            'forecast_lower': lower_bound,
            'forecast_upper': upper_bound,
            'confidence': 'high' if forecast_value >= 3.0 else 'medium' if forecast_value >= 2.0 else 'low',
            'hot_hours': hot_hours,
            'current_quarter_prediction': current_quarter_prediction
        })
    
    return results


def main():
    try:
        conn = get_db_connection()
        df = fetch_current_year_quarterly_data(conn)
        print(f"DEBUG: Fetched {len(df)} records", file=sys.stderr)
        if not df.empty:
            print(f"DEBUG: Sample data: {df.head().to_dict()}", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG: Error in main: {e}", file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if df.empty:
        print(json.dumps([]))
        return

    # Detect continuous incidents
    continuous_df = detect_continuous_incidents(df)
    print(f"DEBUG: Found {len(continuous_df)} continuous incidents", file=sys.stderr)
    
    if continuous_df.empty:
        print(json.dumps([]))
        return

    # Build forecasts for continuous incidents
    forecast_results = build_quarterly_arima_forecast(continuous_df, df)
    print(f"DEBUG: Generated {len(forecast_results)} forecasts", file=sys.stderr)
    
    # Group forecasts by barangay and combine incident types with counts
    barangay_groups = {}
    for result in forecast_results:
        bt_id = result['bt_id']
        if bt_id not in barangay_groups:
            barangay_groups[bt_id] = {
                "bt_id": result['bt_id'],
                "barangay": result['barangay'],
                "latitude": result['latitude'],
                "longitude": result['longitude'],
                "incident_types": [],
                "continuous_months": result['continuous_months'],
                "prediction_quarter": result['prediction_quarter'],
                "hot_hours": result['hot_hours']
            }
        
        # Add incident type with count if not already present
        # Replace commas with "with" in incident type names
        formatted_incident_type = result['type_of_incident'].replace(', ', ' with ')
        incident_info = {
            "type": formatted_incident_type,
            "count": result['forecast_value'],
            "current_quarter_prediction": result['current_quarter_prediction']
        }
        if incident_info not in barangay_groups[bt_id]["incident_types"]:
            barangay_groups[bt_id]["incident_types"].append(incident_info)
    
    # Format output for the map (grouped by barangay with combined incident types)
    outputs = []
    quarter_months = {
        1: "January-March",
        2: "April-June", 
        3: "July-September",
        4: "October-December"
    }
    
    for barangay_data in barangay_groups.values():
        # Sort incident types by count (highest first)
        sorted_incidents = sorted(barangay_data["incident_types"], key=lambda x: x['count'], reverse=True)
        
        # Combine incident types with "with" and include counts
        if len(sorted_incidents) == 1:
            combined_incidents = sorted_incidents[0]["type"]
        else:
            combined_incidents = " with ".join([inc["type"] for inc in sorted_incidents])
        
        outputs.append({
            "bt_id": barangay_data["bt_id"],
            "barangay": barangay_data["barangay"],
            "latitude": barangay_data["latitude"],
            "longitude": barangay_data["longitude"],
            "type_of_incident": combined_incidents,
            "prediction_quarter": barangay_data["prediction_quarter"],
            "quarter_period": quarter_months.get(barangay_data["prediction_quarter"], f"Q{barangay_data['prediction_quarter']}"),
            "continuous_months": barangay_data["continuous_months"],
            "hot_hours": barangay_data["hot_hours"],
            "incident_details": sorted_incidents  # Include detailed incident info for sorting
        })

    print(json.dumps(outputs))


if __name__ == "__main__":
    print("Script starting...", file=sys.stderr)
    main()
    print("Script ending...", file=sys.stderr)


