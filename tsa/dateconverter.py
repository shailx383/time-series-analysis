import re
from datetime import datetime
import pandas as pd


class DateConverter:
    def __init__(self):
        self.formats = [
            "%Y-%m-%d",  # YYYY-MM-DD
            "%Y/%m/%d",  # YYYY/MM/DD
            "%m-%d-%Y",  # MM-DD-YYYY
            "%m/%d/%Y",  # MM/DD/YYYY
            "%d-%m-%Y",  # DD-MM-YYYY
            "%d/%m/%Y",  # DD/MM/YYYY
        ]
        self.directives_info = {'Code': ['%a',
                                         '%A',
                                         '%w',
                                         '%d',
                                         '%-d',
                                         '%b',
                                         '%B',
                                         '%m',
                                         '%-m',
                                         '%y',
                                         '%Y',
                                         '%H',
                                         '%-H',
                                         '%I',
                                         '%-I',
                                         '%p',
                                         '%M',
                                         '%-M',
                                         '%S',
                                         '%-S',
                                         '%f',
                                         '%z',
                                         '%Z'],
                                'Example': ['Sun',
                                            'Sunday',
                                            '0',
                                            '08',
                                            '8',
                                            'Sep',
                                            'September',
                                            '09',
                                            '9',
                                            '13',
                                            '2013',
                                            '07',
                                            '7',
                                            '07',
                                            '7',
                                            'AM',
                                            '06',
                                            '6',
                                            '05',
                                            '5',
                                            '000000',
                                            '+0000',
                                            'UTC'],
                                'Description': ['Weekday as locale\'s abbreviated name.',
                                                'Weekday as locale\'s full name.',
                                                'Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.',
                                                'Day of the month as a zero-padded decimal number.',
                                                'Day of the month as a decimal number. (Platform speciﬁc)',
                                                'Month as locale\'s abbreviated name.',
                                                'Month as locale\'s full name.',
                                                'Month as a zero-padded decimal number.',
                                                'Month as a decimal number. (Platform speciﬁc)',
                                                'Year without century as a zero-padded decimal number.',
                                                'Year with century as a decimal number.',
                                                'Hour (24-hour clock) as a zero-padded decimal number.',
                                                'Hour (24-hour clock) as a decimal number. (Platform speciﬁc)',
                                                'Hour (12-hour clock) as a zero-padded decimal number.',
                                                'Hour (12-hour clock) as a decimal number. (Platform speciﬁc)',
                                                'Locale\'s equivalent of either AM or PM.',
                                                'Minute as a zero-padded decimal number.',
                                                'Minute as a decimal number. (Platform speciﬁc)',
                                                'Second as a zero-padded decimal number.',
                                                'Second as a decimal number. (Platform speciﬁc)',
                                                'Microsecond as a decimal number, zero-padded to 6 digits.',
                                                'UTC oﬀset in the form ±HHMM[SS[.ﬀﬀﬀ]] (empty string if the object is naive).',
                                                'Time zone name (empty string if the object is naive).']}
        return

    def convert_date(self, date: str, include_time = False, format_str=None):
        if not format_str:
            for date_format in self.formats:
                try:
                    date = datetime.strptime(date, date_format)
                    if not include_time:
                        return date.strftime("%Y-%m-%d")
                    else:
                        return datetime.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

            raise ValueError("Invalid date format")

        else:
            if not include_time:
                return datetime.strptime(date, format_str).strftime("%Y-%m-%d")
            else:
                return datetime.strptime(date, format_str).strftime("%Y-%m-%d %H:%M:%S")

    def add_format(self, date_format: str):
        self.formats.append(date_format)
        
    def directive_df(self):
        return pd.DataFrame(self.directives_info)
