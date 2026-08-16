"""
Rain City Roads - data processing module
Elsa Chao, CSE 163, Summer 2026

Loads the four project datasets (NOAA daily weather at SeaTac, the Fremont
and Spokane St bridge bike counters, and SDOT collisions), turns each one
into daily totals, and joins them into the three combined tables the
analysis uses. Every function here takes a file path or a DataFrame and
returns a DataFrame, so no function depends on anything outside itself.
"""
import pandas as pd

# A dry stretch is this many dry days in a row. The first rainy day after
# a stretch this long is what RQ2 calls a "first rain" day.
DRY_STRETCH_DAYS = 7


def load_weather(path):
    """
    Load the NOAA daily weather file.

    Args:
        path: path to the NOAA CSV, which needs DATE, PRCP, TMAX, and
            TMIN columns.

    Returns:
        A DataFrame with one row per day and columns date, PRCP (rain in
        inches), TMAX, and TMIN (degrees F).
    """
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['DATE'])
    df = df[['date', 'PRCP', 'TMAX', 'TMIN']]
    df = df.drop_duplicates(subset='date').sort_values('date')
    return df.reset_index(drop=True)


def load_bridge(path, total_col):
    """
    Load an hourly bike counter file and add the counts up per day.

    One of the counter files writes its totals with commas ("1,234"),
    which pandas reads as text, so the column is turned into a number
    first and anything unreadable becomes missing.

    Args:
        path: path to the counter CSV.
        total_col: name of the column with the combined hourly total.

    Returns:
        A DataFrame with one row per day and columns date and riders.
    """
    df = pd.read_csv(path)
    df[total_col] = pd.to_numeric(
        df[total_col].astype(str).str.replace(',', ''), errors='coerce')
    df['date'] = pd.to_datetime(df['Date'], format='mixed').dt.normalize()
    daily = df.groupby('date')[total_col].sum().reset_index()
    daily = daily.rename(columns={total_col: 'riders'})
    return daily


def load_collisions(path):
    """
    Load the SDOT collisions and count them per day.

    A collision counts as bike-involved if any cyclist was involved, and
    car-only is everything else. The two groups are kept separate on
    purpose: RQ4 needs categories that don't overlap, so bike crashes
    can't also be counted inside the total.

    Args:
        path: path to the SDOT collisions CSV.

    Returns:
        A DataFrame with one row per day and columns date, crashes,
        bike_crashes, and car_only_crashes.
    """
    df = pd.read_csv(path, low_memory=False)
    df['date'] = pd.to_datetime(df['INCDATE'], format='mixed').dt.normalize()
    df['bike'] = df['PEDCYLCOUNT'] > 0
    crashes = df.groupby('date').size().reset_index(name='crashes')
    bikes = df.groupby('date')['bike'].sum().reset_index(name='bike_crashes')
    daily = crashes.merge(bikes, on='date')
    daily['bike_crashes'] = daily['bike_crashes'].astype(int)
    daily['car_only_crashes'] = daily['crashes'] - daily['bike_crashes']
    return daily


def add_rain_flags(weather, dry_stretch_days=DRY_STRETCH_DAYS):
    """
    Mark each day as rainy, and mark the first rain after a dry stretch.

    This walks through the days in order and keeps a running count of how
    many dry days have gone by. If the record skips a date, or a day has
    no rain measurement at all, the count restarts, because a gap in the
    data is not the same thing as a stretch of dry weather.

    Args:
        weather: daily weather table from load_weather.
        dry_stretch_days: how many dry days in a row have to come before
            a rainy day for it to count as a first rain.

    Returns:
        The same table with two new boolean columns, is_rainy and
        is_first_rain.
    """
    weather = weather.sort_values('date').reset_index(drop=True)
    weather['is_rainy'] = weather['PRCP'] > 0

    dates = weather['date'].tolist()
    precip = weather['PRCP'].tolist()
    first_rain = []
    dry_count = 0
    for i in range(len(weather)):
        if i > 0 and (dates[i] - dates[i - 1]).days != 1:
            dry_count = 0
        if pd.isna(precip[i]):
            first_rain.append(False)
            dry_count = 0
        elif precip[i] > 0:
            first_rain.append(dry_count >= dry_stretch_days)
            dry_count = 0
        else:
            first_rain.append(False)
            dry_count = dry_count + 1

    weather['is_first_rain'] = first_rain
    return weather


def combine_bridges(fremont, spokane):
    """
    Add the two bridge counters together into one daily rider total.

    Only days both counters reported are kept, so one bridge never
    stands in for two.

    Args:
        fremont: daily table from load_bridge for the Fremont counter.
        spokane: daily table from load_bridge for the Spokane St counter.

    Returns:
        A DataFrame with columns date, riders_fremont, riders_spokane,
        and riders (the two added together).
    """
    bikes = fremont.merge(spokane, on='date',
                          suffixes=('_fremont', '_spokane'))
    bikes['riders'] = bikes['riders_fremont'] + bikes['riders_spokane']
    return bikes


def build_tables(weather, bikes, collisions):
    """
    Join the daily tables into the three combinations the analysis needs.

    Args:
        weather: daily weather table, already run through add_rain_flags.
        bikes: combined daily rider table from combine_bridges.
        collisions: daily collision table from load_collisions.

    Returns:
        A tuple (weather_collisions, weather_bikes, full). The first
        answers RQ1 and RQ2, the second answers RQ3, and the third has
        everything RQ4 needs.
    """
    weather_collisions = weather.merge(collisions, on='date')
    weather_bikes = weather.merge(bikes[['date', 'riders']], on='date')
    full = weather_collisions.merge(bikes[['date', 'riders']], on='date')
    return weather_collisions, weather_bikes, full


def load_all(weather_path, fremont_path, fremont_col, spokane_path,
             spokane_col, collisions_path):
    """
    Run the whole loading process from files to joined tables.

    Args:
        weather_path: path to the NOAA CSV.
        fremont_path: path to the Fremont counter CSV.
        fremont_col: name of the total column in the Fremont file.
        spokane_path: path to the Spokane St counter CSV.
        spokane_col: name of the total column in the Spokane St file.
        collisions_path: path to the SDOT collisions CSV.

    Returns:
        The same three joined tables that build_tables returns.
    """
    weather = add_rain_flags(load_weather(weather_path))
    bikes = combine_bridges(
        load_bridge(fremont_path, fremont_col),
        load_bridge(spokane_path, spokane_col))
    collisions = load_collisions(collisions_path)
    return build_tables(weather, bikes, collisions)
