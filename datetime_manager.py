"""
Centralized Date/Time Management for Powerball Insights
------------------------------------------------------
Provides timezone-aware datetime handling and consistent date format processing
across all system components.
"""

import datetime
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DateTimeManager:
    """Centralized date/time processing for the entire application."""
    
    UTC_ZONE = ZoneInfo('UTC')
    
    @staticmethod
    def get_utc_timestamp() -> str:
        """Generate timezone-aware UTC timestamp in ISO 8601 format."""
        return datetime.datetime.now(DateTimeManager.UTC_ZONE).isoformat()
    
    @staticmethod
    def get_formatted_timestamp() -> str:
        """Generate timezone-aware UTC timestamp for database storage."""
        return datetime.datetime.now(DateTimeManager.UTC_ZONE).strftime('%Y-%m-%d %H:%M:%S+00:00')
    
    @staticmethod
    def get_csv_date() -> str:
        """Generate current date in CSV format (YYYY-MM-DD)."""
        return datetime.datetime.now(DateTimeManager.UTC_ZONE).strftime('%Y-%m-%d')
    
    @staticmethod
    def parse_date_input(date_input: Any) -> Optional[datetime.date]:
        """
        Parse various date input formats to standard date object.
        
        Args:
            date_input: Date in various formats (string, datetime, date)
            
        Returns:
            datetime.date object or None if parsing fails
        """
        if date_input is None:
            return None
            
        try:
            # Handle datetime.date objects
            if isinstance(date_input, datetime.date):
                return date_input
            
            # Handle datetime.datetime objects
            if isinstance(date_input, datetime.datetime):
                return date_input.date()
            
            # Handle string inputs
            if isinstance(date_input, str):
                # Try YYYY-MM-DD format first
                try:
                    return datetime.datetime.strptime(date_input, '%Y-%m-%d').date()
                except ValueError:
                    pass
                
                # Try MM/DD/YYYY format
                try:
                    return datetime.datetime.strptime(date_input, '%m/%d/%Y').date()
                except ValueError:
                    pass
                
                # Try ISO format
                try:
                    return datetime.datetime.fromisoformat(date_input.replace('Z', '+00:00')).date()
                except ValueError:
                    pass
                
                # Try pandas to_datetime as last resort
                try:
                    return pd.to_datetime(date_input).date()
                except Exception:
                    pass
            
            logger.warning(f"Could not parse date input: {date_input}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing date input {date_input}: {e}")
            return None
    
    @staticmethod
    def format_for_csv(date_obj: Union[datetime.date, datetime.datetime]) -> str:
        """
        Format date for CSV storage (YYYY-MM-DD).
        
        Args:
            date_obj: Date or datetime object
            
        Returns:
            Date string in YYYY-MM-DD format
        """
        if isinstance(date_obj, datetime.datetime):
            return date_obj.strftime('%Y-%m-%d')
        elif isinstance(date_obj, datetime.date):
            return date_obj.strftime('%Y-%m-%d')
        else:
            raise ValueError(f"Invalid date object type: {type(date_obj)}")
    
    @staticmethod
    def format_for_database(dt: Optional[datetime.datetime] = None) -> str:
        """
        Format datetime for database storage (ISO 8601 with timezone).
        
        Args:
            dt: Datetime object, or None for current time
            
        Returns:
            ISO 8601 formatted string with timezone
        """
        if dt is None:
            dt = datetime.datetime.now(DateTimeManager.UTC_ZONE)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=DateTimeManager.UTC_ZONE)
        
        return dt.isoformat()
    
    @staticmethod
    def validate_date_range(date_obj: datetime.date) -> bool:
        """
        Validate date is within acceptable range for lottery data.
        
        Args:
            date_obj: Date to validate
            
        Returns:
            True if date is valid for lottery data
        """
        if date_obj is None:
            return False
        
        # Powerball started in 1992
        earliest_date = datetime.date(1992, 1, 1)
        
        # No future dates beyond 1 year
        latest_date = datetime.datetime.now(DateTimeManager.UTC_ZONE).date() + datetime.timedelta(days=365)
        
        return earliest_date <= date_obj <= latest_date
    
    @staticmethod
    def parse_timestamp(timestamp_str: str) -> Optional[datetime.datetime]:
        """
        Parse various timestamp formats to timezone-aware datetime.
        
        Args:
            timestamp_str: Timestamp string in various formats
            
        Returns:
            Timezone-aware datetime object or None if parsing fails
        """
        if not timestamp_str:
            return None
        
        try:
            # Handle ISO format with timezone
            if '+' in timestamp_str or timestamp_str.endswith('Z'):
                return datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Handle ISO format without timezone (assume UTC)
            try:
                dt = datetime.datetime.fromisoformat(timestamp_str)
                return dt.replace(tzinfo=DateTimeManager.UTC_ZONE)
            except ValueError:
                pass
            
            # Handle standard database format
            try:
                dt = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                return dt.replace(tzinfo=DateTimeManager.UTC_ZONE)
            except ValueError:
                pass
            
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
            return None
    
    @staticmethod
    def convert_naive_to_utc(dt: datetime.datetime) -> datetime.datetime:
        """
        Convert naive datetime to UTC timezone-aware datetime.
        
        Args:
            dt: Naive datetime object
            
        Returns:
            UTC timezone-aware datetime
        """
        if dt.tzinfo is not None:
            return dt.astimezone(DateTimeManager.UTC_ZONE)
        else:
            return dt.replace(tzinfo=DateTimeManager.UTC_ZONE)
    
    @staticmethod
    def standardize_dataframe_dates(df: pd.DataFrame, date_column: str = 'draw_date') -> pd.DataFrame:
        """
        Standardize date column in DataFrame to YYYY-MM-DD format.
        
        Args:
            df: DataFrame with date column
            date_column: Name of the date column
            
        Returns:
            DataFrame with standardized date column
        """
        if df.empty or date_column not in df.columns:
            return df
        
        try:
            # Convert to datetime then to string in standard format
            df[date_column] = pd.to_datetime(df[date_column]).dt.strftime('%Y-%m-%d')
            return df
        except Exception as e:
            logger.error(f"Error standardizing dates in DataFrame: {e}")
            return df

# Global instance for easy access
datetime_manager = DateTimeManager()