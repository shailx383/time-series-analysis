import re
from datetime import datetime
import pandas as pd


class DateConverter:
    '''
        class for conversion of dates of any format into unified pandas datetime object
    '''

    def __init__(self):
        self.formats = [
            "%Y-%m-%d",  # YYYY-MM-DD
            "%Y/%m/%d",  # YYYY/MM/DD
            "%m-%d-%Y",  # MM-DD-YYYY
            "%m/%d/%Y",  # MM/DD/YYYY
            "%d-%m-%Y",  # DD-MM-YYYY
            "%d/%m/%Y",  # DD/MM/YYYY
            "%Y-%m"
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

    def convert_date(self, date: str, include_time=False, format_str=None, infer_format=False):
        '''
            converts date string into unified datetime format

            Args:   
                - date (str): date string
                - include_time (bool): True if output should include time along with date
                - format_str (str): format of input date string, if None infers format from self.formats
                - infer_format (bool): format of input date string, if True infers format based on pandas function pd.to_datetime()

            Returns:
                (str): date in YYYY-MM-DD format
        '''
        if infer_format:
            return pd.to_datetime(date)
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
                date = datetime.strptime(date, format_str).strftime("%Y-%m-%d")
                return pd.to_datetime(date)
            else:
                date = datetime.strptime(
                    date, format_str).strftime("%Y-%m-%d %H:%M:%S")
                return pd.to_datetime(date)

    def add_format(self, date_format: str):
        '''
            adds a date format to the set of date_formats

            Args:
                - date_format (str): date format to be added

            Returns:
                - None
        '''
        self.formats.append(date_format)

    def directive_df(self):
        '''
            dataframe of string directives to specify format

            Args:
                - no arguments

            Returns:
                - (pd.DataFrame): dataframe with all directives, examples, and description
        '''
        return pd.DataFrame(self.directives_info)
