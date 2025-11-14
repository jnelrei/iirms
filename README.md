# Crime Forecast API - Flask Application

This Flask application provides REST API endpoints for crime forecasting and analysis.

## Features

- **Quarterly Forecasts**: Predict crime incidents by quarter based on historical patterns
- **Monthly Forecasts**: Predict crime incidents by month based on last year's data
- **Hot Hours Analysis**: Identify peak times for incidents
- **Geographic Analysis**: Crime predictions by barangay location
- **Statistical Reports**: Incident type counts, top hotspots, and yearly comparisons

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Development Mode
```bash
python app.py
```

The application will start on `http://localhost:5000`

### Production Mode
Use a WSGI server like Gunicorn:
```bash
gunicorn app:app
```

## API Endpoints

### 1. Home
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Check if the API is running and database is connected.

### 3. Quarterly Forecast
```
GET /forecast/quarterly
```
Returns quarterly crime forecasts with continuous incident detection.

**Response Format:**
```json
{
  "forecasts": [
    {
      "bt_id": 1,
      "barangay": "Barangay Name",
      "latitude": 14.123,
      "longitude": 121.456,
      "type_of_incident": "Incident Type",
      "prediction_quarter": 2,
      "quarter_period": "April-June",
      "continuous_months": "Q1(Jan,Feb,Mar)",
      "hot_hours": "8:00 AM – 5:00 PM",
      "incident_details": [...]
    }
  ],
  "statistics": {
    "incident_type_counts": {...},
    "top_hotspots": [...],
    "yearly_comparison": [...]
  }
}
```

### 4. Monthly Forecast
```
GET /forecast/monthly?target_month=1
```
Returns monthly crime forecasts based on last year's data.

**Query Parameters:**
- `target_month` (optional): Integer 1-12 for specific month forecast. If omitted, returns all months.

**Response Format:**
```json
[
  {
    "bt_id": 1,
    "barangay": "Barangay Name",
    "latitude": 14.123,
    "longitude": 121.456,
    "type_of_incident": "Incident Type",
    "prediction_quarter": 1,
    "quarter_period": "January",
    "continuous_months": "January",
    "hot_hours": "8:00 AM – 5:00 PM",
    "incident_details": [...]
  }
]
```

## Usage Examples

### Get Quarterly Forecast
```bash
curl http://localhost:5000/forecast/quarterly
```

### Get Monthly Forecast for January
```bash
curl http://localhost:5000/forecast/monthly?target_month=1
```

### Get All Monthly Forecasts
```bash
curl http://localhost:5000/forecast/monthly
```

## Configuration

The database connection settings can be modified in the `get_db_connection()` function within `app.py`:

```python
def get_db_connection():
    return pymysql.connect(
        host="your_host",
        user="your_user",
        password="your_password",
        database="your_database",
        cursorclass=pymysql.cursors.DictCursor,
    )
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Internal Server Error

Error responses follow this format:
```json
{
  "error": "Error message description"
}
```

## Logging

The application uses Flask's built-in logger. Debug logs are enabled in development mode.

## Migration from CLI Script

This Flask application is converted from a command-line script (`py.py`). The main differences:

- **CLI**: Run with arguments `python py.py quarterly` or `python py.py monthly 1`
- **Flask**: Use HTTP endpoints `/forecast/quarterly` or `/forecast/monthly?target_month=1`

All forecasting logic, database queries, and data processing remain identical.

