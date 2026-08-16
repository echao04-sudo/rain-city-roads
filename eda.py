"""
Rain City Roads: EDA
Elsa Chao, CSE 163
Loads collision, bike counter, and weather data,
aggregates everything to daily totals, and joins them.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def load_weather(path):
    """Load NOAA daily weather, keep date, precip, temps."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['DATE'])
    df = df[['date', 'PRCP', 'TMAX', 'TMIN']]
    return df


def load_bridge(path, total_col):
    """Load an hourly bridge counter file and sum counts per day."""
    df = pd.read_csv(path)
    df[total_col] = pd.to_numeric(
        df[total_col].astype(str).str.replace(',', ''),
        errors='coerce')
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


def summarize(name, df):
    """Print size, missingness, and summary stats for a table."""
    print('=====', name, '=====')
    print('shape:', df.shape)
    print('missing values per column:')
    print(df.isnull().sum())
    print('summary of variables:')
    print(df.describe().to_string())
    print()


def plot_rainy_vs_dry(wc):
    """Bar chart: average daily crashes, rainy vs dry days."""
    wc = wc.dropna(subset=['PRCP'])
    wc['rainy'] = wc['PRCP'] > 0
    means = wc.groupby('rainy')['crashes'].mean()
    fig, ax = plt.subplots()
    ax.bar(['Dry', 'Rainy'], [means[False], means[True]])
    ax.set_ylabel('Average crashes per day')
    ax.set_title('Average Daily Collisions: Rainy vs Dry Days')
    fig.savefig('plots/rainy_vs_dry.png', bbox_inches='tight')
    plt.close(fig)


def plot_scatter(df, xcol, ycol, xlabel, ylabel, title, fname):
    """Generic scatter plot saved to plots/."""
    fig, ax = plt.subplots()
    ax.scatter(df[xcol], df[ycol], s=5, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig('plots/' + fname, bbox_inches='tight')
    plt.close(fig)


def check_assumptions(wc, full):
    """Verify assumptions for planned statistical tests."""
    wc = wc.dropna(subset=['PRCP'])
    rainy = wc[wc['PRCP'] > 0]['crashes']
    dry = wc[wc['PRCP'] == 0]['crashes']
    print('t-test groups: rainy n =', len(rainy),
          'dry n =', len(dry))
    print('rainy mean/std:', rainy.mean(), rainy.std())
    print('dry mean/std:', dry.mean(), dry.std())
    full = full.dropna(subset=['PRCP'])
    full['rainy'] = full['PRCP'] > 0
    table = full.groupby('rainy')[['bike_crashes', 'crashes']].sum()
    print('chi-square table (check counts >= 5):')
    print(table)


def main():
    weather = load_weather('data/NOAA.csv')
    fremont = load_bridge(
        'data/Fremont_Bridge.csv',
        'Fremont Bridge Sidewalks, south of N 34th St Total')
    spokane = load_bridge('data/Spokane_St.csv',
                          'Spokane St. Bridge Total')
    collisions = load_collisions('data/SDOT.csv')

    wc, wb, full = build_tables(weather, fremont, spokane, collisions)

    summarize('weather + collisions', wc)
    summarize('weather + bikes', wb)
    summarize('full (all four)', full)

    plot_rainy_vs_dry(wc)
    plot_scatter(wc, 'PRCP', 'crashes', 'Precipitation (in)',
                 'Crashes per day', 'Precipitation vs Daily Collisions',
                 'prcp_vs_crashes.png')
    plot_scatter(wb, 'PRCP', 'riders', 'Precipitation (in)',
                 'Bridge crossings per day',
                 'Precipitation vs Daily Bike Ridership',
                 'prcp_vs_riders.png')
    plot_scatter(wb, 'TMAX', 'riders', 'High temperature (F)',
                 'Bridge crossings per day',
                 'Temperature vs Daily Bike Ridership',
                 'tmax_vs_riders.png')
    check_assumptions(wc, full)


if __name__ == '__main__':
    main()
