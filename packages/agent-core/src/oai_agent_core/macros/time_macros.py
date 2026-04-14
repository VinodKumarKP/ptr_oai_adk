from datetime import datetime, timedelta

def macro_date(*args, **kwargs):
    """
    Macro to return formatted date.
    Usage: {{ DATE [format] [offset_days] }}
    Default format: %Y-%m-%d
    Default offset: 0 (today)
    """
    fmt = "%Y-%m-%d"
    offset = 0

    if len(args) > 0:
        fmt = args[0]
    
    if len(args) > 1:
        try:
            offset = int(args[1])
        except ValueError:
            return f"Error: Invalid offset '{args[1]}'. Must be an integer."

    target_date = datetime.now() + timedelta(days=offset)
    return target_date.strftime(fmt)

def macro_now(*args, **kwargs):
    """
    Macro to return current timestamp.
    Usage: {{ NOW [format] }}
    Default format: ISO 8601
    """
    if args:
        return datetime.now().strftime(args[0])
    return datetime.now().isoformat()

def macro_timestamp(*args, **kwargs):
    """
    Macro to get Unix timestamp.
    Usage: {{ TIMESTAMP [offset_seconds] [format] }}
    
    offset_seconds: Number of seconds to add/subtract (optional)
    format: If 'iso', returns ISO 8601 format; if 'ms', returns milliseconds (optional)
    
    Examples:
    {{ TIMESTAMP }} - Current Unix timestamp
    {{ TIMESTAMP -3600 }} - One hour ago
    {{ TIMESTAMP 0 iso }} - Current time in ISO format
    {{ TIMESTAMP 0 ms }} - Current time in milliseconds
    """
    offset_seconds = 0
    output_format = 'unix'
    
    if len(args) > 0:
        try:
            offset_seconds = int(args[0])
        except ValueError:
            return "Error: offset_seconds must be an integer."
    
    if len(args) > 1:
        output_format = args[1].lower()
    
    try:
        dt = datetime.now() + timedelta(seconds=offset_seconds)
        
        if output_format == 'iso':
            return dt.isoformat()
        elif output_format == 'ms':
            return str(int(dt.timestamp() * 1000))
        else:
            return str(int(dt.timestamp()))
    except Exception as e:
        return f"Error generating timestamp: {e}"

def macro_business_days_from(*args, **kwargs):
    """
    Macro to calculate business days from a date.
    Usage: {{ BUSINESS_DAYS_FROM date count [format] }}
    
    date: Starting date (YYYY-MM-DD format or 'today')
    count: Number of business days to add (can be negative)
    format: Output format (default: %Y-%m-%d)
    
    Examples:
    {{ BUSINESS_DAYS_FROM today 5 }} - 5 business days from today
    {{ BUSINESS_DAYS_FROM 2024-01-15 -3 }} - 3 business days before Jan 15
    """
    if len(args) < 2:
        return "Error: BUSINESS_DAYS_FROM macro requires date and count."
    
    date_str = args[0]
    try:
        count = int(args[1])
    except ValueError:
        return "Error: count must be an integer."
    
    output_format = args[2] if len(args) > 2 else "%Y-%m-%d"
    
    try:
        # Parse input date
        if date_str.lower() == 'today':
            current_date = datetime.now().date()
        else:
            current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Calculate business days
        days_added = 0
        direction = 1 if count > 0 else -1
        target_days = abs(count)
        
        while days_added < target_days:
            current_date += timedelta(days=direction)
            # Skip weekends (Monday=0, Sunday=6)
            if current_date.weekday() < 5:
                days_added += 1
        
        return current_date.strftime(output_format)
        
    except ValueError as e:
        return f"Error: Invalid date format. Use YYYY-MM-DD or 'today'. {e}"
    except Exception as e:
        return f"Error calculating business days: {e}"
