"""
Rain City Roads - EDA
Elsa Chao, CSE 163
Loads collision, bike counter, and weather data,
aggregates everything to daily totals, and joins them.
"""
import pandas as pd


def load_weather(path):
    """Load NOAA daily weather, keep date, precip, temps."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['DATE'])
    df = df[['date', 'PRCP', 'TMAX', 'TMIN']]
    return df


def load_bridge(path, total_col):
    """Load an hourly bridge counter file and sum counts per day."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(
        df['Date'], format='mixed').dt.normalize()
    daily = df.groupby('date')[total_col].sum().reset_index()
    daily = daily.rename(columns={total_col: 'riders'})
    return daily


def load_collisions(path):
    """Load collisions, count total + bike-involved crashes per day."""
    df = pd.read_csv(path, low_memory=False)
    df['date'] = pd.to_datetime(
        df['INCDATE'], format='mixed').dt.normalize()
    df['bike'] = df['PEDCYLCOUNT'] > 0
    crashes = df.groupby('date').size().reset_index(name='crashes')
    bikes = df.groupby('date')['bike'].sum().reset_index(
        name='bike_crashes')
    daily = crashes.merge(bikes, on='date')
    return daily


def build_tables(weather, fremont, spokane, collisions):
    """Join daily tables into the combos needed for each RQ."""
    # total riders across both bridges (RQ3, RQ4)
    bikes = fremont.merge(spokane, on='date',
                          suffixes=('_fremont', '_spokane'))
    bikes['riders'] = (bikes['riders_fremont']
                       + bikes['riders_spokane'])
    # weather + collisions (RQ1, RQ2)
    wc = weather.merge(collisions, on='date')
    # weather + bikes (RQ3)
    wb = weather.merge(bikes[['date', 'riders']], on='date')
    # everything (RQ4)
    full = wc.merge(bikes[['date', 'riders']], on='date')
    return wc, wb, full


def main():
    weather = load_weather('data/NOAA.csv')
    fremont = load_bridge(
        'data/Fremont_Bridge.csv',
        'Fremont Bridge Sidewalks, south of N 34th St Total')
    spokane = load_bridge('data/Spokane_St.csv',
                          'Spokane St. Bridge Total')
    collisions = load_collisions('data/SDOT.csv')

    print('weather:', weather.shape)
    print('fremont:', fremont.shape)
    print('spokane:', spokane.shape)
    print('collisions:', collisions.shape)

    wc, wb, full = build_tables(weather, fremont, spokane, collisions)
    print('weather+collisions:', wc.shape)
    print('weather+bikes:', wb.shape)
    print('full:', full.shape)


if __name__ == '__main__':
    main()