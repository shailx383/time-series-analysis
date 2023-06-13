import re
from datetime import datetime

def convert_to_unified_format(date_string):
    formats = [
        "%Y-%m-%d",  # YYYY-MM-DD
        "%Y/%m/%d",  # YYYY/MM/DD
        "%m-%d-%Y",  # MM-DD-YYYY
        "%m/%d/%Y",  # MM/DD/YYYY
        "%d-%m-%Y",  # DD-MM-YYYY
        "%d/%m/%Y",  # DD/MM/YYYY
    ]

    for date_format in formats:
        try:
            date = datetime.strptime(date_string, date_format)
            return date.strftime("%Y-%m-%d")
        except ValueError:
            pass

    raise ValueError("Invalid date format")

# Example usage
date_input = input("Enter a date: ")
unified_date = convert_to_unified_format(date_input)
print("Unified date format:", unified_date)
