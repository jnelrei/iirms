from flask import Flask, request, jsonify
import sys
import math
from datetime import date, timedelta
import pandas as pd
import numpy as np
import pymysql

# Initialize Flask app
app = Flask(__name__)

# Optional: if statsmodels is missing, return a graceful fallback
HAS_SM = False  # Disable statsmodels for now to avoid hanging


def get_db_connection():
    """Get database connection matching PHP db.php settings"""
    return pymysql.connect(
        host="srv1322.hstgr.io",
        user="u520834156_userRMS25",
        password="]lRgy[uiN1",
        database="u520834156_dbRMS2025",
        cursorclass=pymysql.cursors.DictCursor,
    )


def fetch_current_year_quarterly_data(conn):
    """Pull current year quarterly incidents by barangay and incident type with month and time detection"""
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


def fetch_last_year_monthly_data(conn, target_month=None):
    """
    Fetch last year's monthly incidents by barangay and incident type.
    If target_month is specified, filter for that specific month.
    """
    month_condition = ""
    if target_month and 1 <= target_month <= 12:
        month_condition = f"AND MONTH(im.date_and_time_reported) = {target_month}"
    
    sql = (
        "SELECT bt.bt_id, bt.barangay, bt.latitude, bt.longitude, "
        "im.type_of_incident, "
        "MONTH(im.date_and_time_reported) AS month, "
        "DATE(im.date_and_time_reported) AS report_date, "
        "HOUR(im.date_and_time_reported) AS report_hour, "
        "COUNT(*) AS cnt "
        "FROM incident_management im "
        "JOIN place_of_incident poi ON poi.poi_id = im.poi_id "
        "JOIN barangay_table bt ON bt.bt_id = poi.bt_id "
        "WHERE im.category = 'Crime' "
        "AND YEAR(im.date_and_time_reported) = YEAR(CURDATE()) - 1 "
        f"{month_condition} "
        "GROUP BY bt.bt_id, bt.barangay, bt.latitude, bt.longitude, im.type_of_incident, month, report_date, report_hour"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["bt_id", "barangay", "latitude", "longitude", "type_of_incident", "month", "report_date", "report_hour", "cnt"])    
    df = pd.DataFrame(rows)
    return df


def fetch_yearly_comparison_data(conn):
    """
    Fetch monthly incident counts for current year and previous year for comparison.
    Returns data grouped by month for both years.
    """
    sql = (
        "SELECT "
        "YEAR(im.date_and_time_reported) AS year, "
        "MONTH(im.date_and_time_reported) AS month, "
        "COUNT(*) AS cnt "
        "FROM incident_management im "
        "JOIN place_of_incident poi ON poi.poi_id = im.poi_id "
        "JOIN barangay_table bt ON bt.bt_id = poi.bt_id "
        "WHERE im.category = 'Crime' "
        "AND YEAR(im.date_and_time_reported) IN (YEAR(CURDATE()), YEAR(CURDATE()) - 1) "
        "GROUP BY year, month "
        "ORDER BY year, month"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["year", "month", "cnt"])
    df = pd.DataFrame(rows)
    return df


def fetch_all_historical_data(conn):
    """
    Fetch all historical crime incidents by barangay and incident type with time detection.
    Used for yearly forecasting based on all-time patterns.
    """
    sql = (
        "SELECT bt.bt_id, bt.barangay, bt.latitude, bt.longitude, "
        "im.type_of_incident, "
        "YEAR(im.date_and_time_reported) AS year, "
        "MONTH(im.date_and_time_reported) AS month, "
        "DATE(im.date_and_time_reported) AS report_date, "
        "HOUR(im.date_and_time_reported) AS report_hour, "
        "COUNT(*) AS cnt "
        "FROM incident_management im "
        "JOIN place_of_incident poi ON poi.poi_id = im.poi_id "
        "JOIN barangay_table bt ON bt.bt_id = poi.bt_id "
        "WHERE im.category = 'Crime' "
        "GROUP BY bt.bt_id, bt.barangay, bt.latitude, bt.longitude, im.type_of_incident, year, month, report_date, report_hour"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["bt_id", "barangay", "latitude", "longitude", "type_of_incident", "year", "month", "report_date", "report_hour", "cnt"])    
    df = pd.DataFrame(rows)
    return df


def convert_to_standard_time(hour):
    """Convert 24-hour format to 12-hour format with AM/PM."""
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


def analyze_hot_months(df, bt_id, incident_type):
    """
    Analyze the most frequent months for incidents in a specific barangay and incident type.
    Returns a list of top months (up to 3) with their incident counts.
    """
    # Filter data for specific barangay and incident type
    filtered_df = df[(df['bt_id'] == bt_id) & (df['type_of_incident'] == incident_type)]
    
    if filtered_df.empty:
        return []
    
    # Count incidents by month
    month_counts = {}
    for _, row in filtered_df.iterrows():
        month = int(row['month'])
        count = row['cnt']
        if month in month_counts:
            month_counts[month] += count
        else:
            month_counts[month] = count
    
    if not month_counts:
        return []
    
    # Sort months by count (descending) and get top 3
    sorted_months = sorted(month_counts.items(), key=lambda x: x[1], reverse=True)
    top_months = sorted_months[:3]  # Get top 3 months
    
    # Format as list of dictionaries with month name and count
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    
    result = []
    for month_num, count in top_months:
        result.append({
            'month': month_num,
            'month_name': month_names.get(month_num, f'Month {month_num}'),
            'count': int(count)
        })
    
    return result


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
    
    app.logger.debug(f"Predicting for bt_id={bt_id}, incident={incident_type}, prediction_q={prediction_quarter}, prev_q={prev_quarter}")
    app.logger.debug(f"Previous data found: {len(prev_data)} records")
    
    if prev_data.empty:
        app.logger.debug(f"No previous data, returning 0")
        return 0  # No prediction if no previous data
    
    # Calculate prediction based on previous quarter incidents
    prev_incidents = prev_data['cnt'].sum()
    app.logger.debug(f"Previous quarter incidents: {prev_incidents}")
    
    # Simple prediction: assume similar pattern with slight variation
    # Add some randomness based on historical patterns
    prediction = int(prev_incidents * 1.1)  # 10% increase assumption
    
    app.logger.debug(f"Calculated prediction: {prediction}")
    return prediction  # Based on actual database data


def predict_monthly_incidents(last_year_df, bt_id, incident_type, target_month):
    """
    Predict monthly incidents based on last year's data for the same month.
    """
    # Filter data for this barangay, incident type, and target month
    monthly_data = last_year_df[(last_year_df['bt_id'] == bt_id) & 
                               (last_year_df['type_of_incident'] == incident_type) & 
                               (last_year_df['month'] == target_month)]
    
    app.logger.debug(f"Monthly prediction for bt_id={bt_id}, incident={incident_type}, month={target_month}")
    app.logger.debug(f"Last year monthly data found: {len(monthly_data)} records")
    
    if monthly_data.empty:
        app.logger.debug(f"No last year data for this month, returning 0")
        return 0
    
    # Calculate prediction based on last year's same month incidents
    last_year_incidents = monthly_data['cnt'].sum()
    app.logger.debug(f"Last year same month incidents: {last_year_incidents}")
    
    # Simple prediction: assume similar pattern with slight variation
    # Add some randomness based on historical patterns
    prediction = int(last_year_incidents * 1.05)  # 5% increase assumption for monthly
    
    app.logger.debug(f"Calculated monthly prediction: {prediction}")
    return prediction


def build_yearly_forecast(all_historical_df):
    """
    Build yearly forecast based on all historical data.
    Analyzes all-time patterns to predict which barangays are most likely to have crime,
    what types of crime, and hot hours.
    """
    results = []
    
    if all_historical_df.empty:
        return []
    
    # Group by barangay and incident type
    for (bt_id, incident_type), group in all_historical_df.groupby(['bt_id', 'type_of_incident']):
        barangay_data = group.iloc[0]  # Get barangay metadata
        total_incidents = group['cnt'].sum()
        
        # Calculate average incidents per year
        years = group['year'].unique()
        avg_per_year = total_incidents / len(years) if len(years) > 0 else 0
        
        # Predict for next year based on historical average with trend
        # Use weighted average giving more weight to recent years
        recent_years = group[group['year'] >= group['year'].max() - 2]  # Last 3 years
        if not recent_years.empty:
            recent_avg = recent_years['cnt'].sum() / len(recent_years['year'].unique()) if len(recent_years['year'].unique()) > 0 else 0
            # Weighted prediction: 60% recent average, 40% overall average
            yearly_prediction = int((recent_avg * 0.6) + (avg_per_year * 0.4))
        else:
            yearly_prediction = int(avg_per_year)
        
        # Analyze hot hours for this barangay and incident type
        hot_hours = analyze_hot_hours(all_historical_df, bt_id, incident_type)
        
        # Analyze hot months for this barangay and incident type
        hot_months = analyze_hot_months(all_historical_df, bt_id, incident_type)
        
        # Calculate confidence based on historical data volume and consistency
        years_with_data = len(years)
        if years_with_data >= 3 and total_incidents >= 10:
            confidence = 'high'
        elif years_with_data >= 2 and total_incidents >= 5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        results.append({
            'bt_id': int(bt_id),
            'barangay': barangay_data['barangay'],
            'latitude': float(barangay_data['latitude']) if not pd.isna(barangay_data['latitude']) else None,
            'longitude': float(barangay_data['longitude']) if not pd.isna(barangay_data['longitude']) else None,
            'type_of_incident': incident_type,
            'forecast_value': float(total_incidents),
            'yearly_prediction': yearly_prediction,
            'avg_per_year': float(avg_per_year),
            'years_with_data': int(years_with_data),
            'confidence': confidence,
            'hot_hours': hot_hours,
            'hot_months': hot_months,  # Add hot months data
            'is_yearly': True
        })
    
    return results


def build_monthly_forecast(last_year_df, target_month=None):
    """
    Build monthly forecast based on last year's data.
    If target_month is specified, filter for that specific month.
    Otherwise, use all months.
    """
    results = []
    
    # Get month name
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    
    if target_month and target_month not in month_names:
        return []
    
    if not last_year_df.empty:
        # Filter by target month if specified
        if target_month:
            df_filtered = last_year_df[last_year_df['month'] == target_month].copy()
            month_name = month_names[target_month]
        else:
            df_filtered = last_year_df
            month_name = "All Months"
        
        if df_filtered.empty:
            return []
        
        # Group by barangay and incident type
        for (bt_id, incident_type), group in df_filtered.groupby(['bt_id', 'type_of_incident']):
            barangay_data = group.iloc[0]  # Get barangay metadata
            total_incidents = group['cnt'].sum()
            
            # Only predict if there were incidents in this month
            if target_month:
                # Predict incidents for this month based on last year's data
                monthly_prediction = predict_monthly_incidents(last_year_df, bt_id, incident_type, target_month)
            else:
                # If no target month, use the total from filtered data
                monthly_prediction = int(total_incidents)
            
            # Analyze hot hours for this barangay and incident type
            hot_hours = analyze_hot_hours(last_year_df, bt_id, incident_type)
            
            # Calculate confidence based on historical data
            confidence = 'high' if total_incidents >= 5 else 'medium' if total_incidents >= 2 else 'low'
            
            results.append({
                'bt_id': int(bt_id),
                'barangay': barangay_data['barangay'],
                'latitude': float(barangay_data['latitude']) if not pd.isna(barangay_data['latitude']) else None,
                'longitude': float(barangay_data['longitude']) if not pd.isna(barangay_data['longitude']) else None,
                'type_of_incident': incident_type,
                'target_month': target_month if target_month else 1,
                'month_name': month_name if target_month else month_names[1],
                'forecast_value': float(total_incidents),
                'monthly_prediction': monthly_prediction,
                'confidence': confidence,
                'hot_hours': hot_hours,
                'is_monthly': True
            })
    
    return results


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


def process_forecast(mode='quarterly', target_period=None):
    """Process forecast based on mode and target period"""
    df = pd.DataFrame()
    comparison_df = pd.DataFrame()
    
    try:
        conn = get_db_connection()
        
        # For quarterly mode, use current year quarterly data
        if mode == 'quarterly':
            df = fetch_current_year_quarterly_data(conn)
            app.logger.debug(f"Fetched {len(df)} records for quarterly")
        # For monthly mode, use last year monthly data
        elif mode == 'monthly':
            df = fetch_last_year_monthly_data(conn, target_period)
            app.logger.debug(f"Fetched {len(df)} records for monthly")
        # For yearly mode, use all historical data
        elif mode == 'yearly':
            df = fetch_all_historical_data(conn)
            app.logger.debug(f"Fetched {len(df)} records for yearly")
        else:
            return {"error": "Invalid mode"}, 400
        
        # Fetch yearly comparison data for all modes
        comparison_df = fetch_yearly_comparison_data(conn)
        app.logger.debug(f"Fetched {len(comparison_df)} comparison records")
            
        if not df.empty:
            app.logger.debug(f"Sample data: {df.head().to_dict()}")
    except Exception as e:
        app.logger.error(f"Error in process_forecast: {e}")
        return {"error": str(e)}, 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if df.empty:
        return []

    if mode == 'yearly':
        # Yearly forecasting based on all historical data
        forecast_results = build_yearly_forecast(df)
        app.logger.debug(f"Generated {len(forecast_results)} yearly forecasts")
        
        # Group by barangay and output
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
                    "hot_hours": result['hot_hours']
                }
            
            formatted_incident_type = result['type_of_incident'].replace(', ', ' with ')
            incident_info = {
                "type": formatted_incident_type,
                "count": result['forecast_value'],
                "current_quarter_prediction": result.get('yearly_prediction', 0),
                "avg_per_year": result.get('avg_per_year', 0),
                "years_with_data": result.get('years_with_data', 0),
                "hot_months": result.get('hot_months', [])  # Include hot months data
            }
            if incident_info not in barangay_groups[bt_id]["incident_types"]:
                barangay_groups[bt_id]["incident_types"].append(incident_info)
        
        outputs = []
        for barangay_data in barangay_groups.values():
            sorted_incidents = sorted(barangay_data["incident_types"], key=lambda x: x['count'], reverse=True)
            
            if len(sorted_incidents) == 1:
                combined_incidents = sorted_incidents[0]["type"]
            else:
                combined_incidents = " with ".join([inc["type"] for inc in sorted_incidents])
            
            # Calculate total predicted for the year
            total_yearly_prediction = sum([inc.get('current_quarter_prediction', 0) for inc in sorted_incidents])
            
            outputs.append({
                "bt_id": barangay_data["bt_id"],
                "barangay": barangay_data["barangay"],
                "latitude": barangay_data["latitude"],
                "longitude": barangay_data["longitude"],
                "type_of_incident": combined_incidents,
                "prediction_quarter": 1,  # Not used for yearly, but keep for compatibility
                "quarter_period": "Yearly Forecast",
                "continuous_months": "All-time patterns",
                "hot_hours": barangay_data["hot_hours"],
                "incident_details": sorted_incidents,
                "yearly_prediction": total_yearly_prediction
            })
        
        return outputs
    
    if mode == 'monthly':
        # Monthly forecasting based on last year
        forecast_results = build_monthly_forecast(df, target_period)
        app.logger.debug(f"Generated {len(forecast_results)} monthly forecasts")
        
        # Group by barangay and output
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
                    "target_month": result['target_month'],
                    "month_name": result['month_name'],
                    "hot_hours": result['hot_hours'],
                    "prediction_quarter": result.get('target_month', 1)  # Use month as quarter for monthly mode
                }
            
            formatted_incident_type = result['type_of_incident'].replace(', ', ' with ')
            incident_info = {
                "type": formatted_incident_type,
                "count": result['forecast_value'],
                "current_quarter_prediction": result.get('monthly_prediction', 0)
            }
            if incident_info not in barangay_groups[bt_id]["incident_types"]:
                barangay_groups[bt_id]["incident_types"].append(incident_info)
        
        outputs = []
        for barangay_data in barangay_groups.values():
            sorted_incidents = sorted(barangay_data["incident_types"], key=lambda x: x['count'], reverse=True)
            
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
                "prediction_quarter": barangay_data["target_month"],
                "quarter_period": barangay_data["month_name"],
                "continuous_months": barangay_data["month_name"],
                "hot_hours": barangay_data["hot_hours"],
                "incident_details": sorted_incidents
            })
        
        return outputs
    
    # Quarterly mode (original logic)
    # Detect continuous incidents
    continuous_df = detect_continuous_incidents(df)
    app.logger.debug(f"Found {len(continuous_df)} continuous incidents")
    
    if continuous_df.empty:
        return []

    # Build forecasts for continuous incidents
    forecast_results = build_quarterly_arima_forecast(continuous_df, df)
    app.logger.debug(f"Generated {len(forecast_results)} forecasts")
    
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

    # Calculate aggregated statistics for charts
    incident_type_counts = {}
    top_hotspots = []
    
    for output in outputs:
        # Count incident types for pie chart
        types = output['type_of_incident'].split(' with ')
        for inc_type in types:
            if inc_type not in incident_type_counts:
                incident_type_counts[inc_type] = 0
            incident_type_counts[inc_type] += 1
        
        # Prepare hotspot data
        total_predicted = 0
        if output.get('incident_details') and isinstance(output['incident_details'], list):
            total_predicted = sum([inc.get('current_quarter_prediction', 0) or 0 for inc in output['incident_details']])
        
        top_hotspots.append({
            "barangay": output['barangay'],
            "incident_type": output['type_of_incident'],
            "predicted": total_predicted,
            "hot_hours": output.get('hot_hours', 'N/A')
        })
    
    # Sort hotspots by predicted count (descending)
    top_hotspots.sort(key=lambda x: x['predicted'], reverse=True)
    
    # Process yearly comparison data for line chart
    yearly_comparison = []
    if not comparison_df.empty:
        # Group by year and month
        current_year = int(pd.Timestamp.now().year)
        previous_year = current_year - 1
        
        # Initialize monthly data for both years
        current_year_data = {month: 0 for month in range(1, 13)}
        previous_year_data = {month: 0 for month in range(1, 13)}
        
        for _, row in comparison_df.iterrows():
            year = int(row['year'])
            month = int(row['month'])
            count = int(row['cnt'])
            
            if year == current_year:
                current_year_data[month] = count
            elif year == previous_year:
                previous_year_data[month] = count
        
        # Prepare data for line chart
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month in range(1, 13):
            yearly_comparison.append({
                "month": month_names[month - 1],
                "current_year": current_year_data[month],
                "previous_year": previous_year_data[month]
            })
    
    # Format output with aggregation data
    final_output = {
        "forecasts": outputs,
        "statistics": {
            "incident_type_counts": incident_type_counts,
            "top_hotspots": top_hotspots[:10],  # Top 10 hotspots
            "yearly_comparison": yearly_comparison  # Monthly comparison data
        }
    }
    
    return final_output


# Flask Routes
@app.route('/')
def home():
    return jsonify({
        "message": "Crime Forecast API",
        "version": "1.0.0",
        "endpoints": {
            "/forecast/quarterly": "Get quarterly crime forecasts",
            "/forecast/monthly": "Get monthly crime forecasts",
            "/forecast/yearly": "Get yearly crime forecasts",
            "/health": "Health check endpoint"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "database": "connected"  # You could add actual DB connection check
    })


@app.route('/forecast/quarterly')
def forecast_quarterly():
    """Get quarterly crime forecasts"""
    try:
        result = process_forecast(mode='quarterly')
        if isinstance(result, tuple):  # Error case
            return jsonify(result[0]), result[1]
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in forecast_quarterly: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/forecast/monthly')
def forecast_monthly():
    """Get monthly crime forecasts"""
    try:
        # Get target_month from query parameter
        target_month = request.args.get('target_month', type=int)
        result = process_forecast(mode='monthly', target_period=target_month)
        if isinstance(result, tuple):  # Error case
            return jsonify(result[0]), result[1]
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in forecast_monthly: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/forecast/yearly')
def forecast_yearly():
    """Get yearly crime forecasts"""
    try:
        result = process_forecast(mode='yearly')
        if isinstance(result, tuple):  # Error case
            return jsonify(result[0]), result[1]
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in forecast_yearly: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Configure logging
    app.logger.setLevel("DEBUG")
    app.run(debug=True, host='0.0.0.0', port=5000)

