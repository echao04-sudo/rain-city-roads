"""
Rain City Roads, analysis module
Elsa Chao, CSE 163, Summer 2026

Answers the four research questions using the joined daily tables built by
data_processing.py, and saves the figures used in the report. Run this file
directly to reproduce every number and plot in the results section.
"""
import os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import data_processing as dp  # noqa: E402

# Where the data and the saved figures live.
DATA_DIR = 'data'
PLOT_DIR = 'plots'
FREMONT_COL = 'Fremont Bridge Sidewalks, south of N 34th St Total'
SPOKANE_COL = 'Spokane St. Bridge Total'

# Colors picked to stay apart for colorblind readers.
DRY_COLOR = '#0072B2'
RAINY_COLOR = '#D55E00'
FIRST_RAIN_COLOR = '#009E73'

# 95% of a normal distribution sits within this many standard errors.
Z_95 = 1.96
CROSSINGS_PER_UNIT = 1000


def confidence_interval(values):
    """
    Find the half-width of a 95% confidence interval for an average.

    Args:
        values: a Series of daily numbers.

    Returns:
        A number to add and subtract from the mean to get the interval.
    """
    values = values.dropna()
    standard_error = values.std() / (len(values) ** 0.5)
    return Z_95 * standard_error


def compare_rainy_dry_crashes(weather_collisions):
    """
    RQ1: are there more collisions on rainy days than dry days?

    Splits the days into rainy (any measurable rain) and dry, then runs a
    two sample t-test on the daily collision counts.

    Args:
        weather_collisions: joined daily weather and collision table.

    Returns:
        A dictionary with the group sizes, averages, standard deviations,
        the percent difference, and the t-test statistic and p-value.
    """
    df = weather_collisions.dropna(subset=['PRCP'])
    rainy = df[df['is_rainy']]['crashes']
    dry = df[~df['is_rainy']]['crashes']
    t_stat, p_value = stats.ttest_ind(rainy, dry)
    return {
        'n_rainy': len(rainy),
        'n_dry': len(dry),
        'mean_rainy': rainy.mean(),
        'mean_dry': dry.mean(),
        'std_rainy': rainy.std(),
        'std_dry': dry.std(),
        'pct_increase': 100 * (rainy.mean() - dry.mean()) / dry.mean(),
        't_stat': t_stat,
        'p_value': p_value,
    }


def compare_first_rain_crashes(weather_collisions):
    """
    RQ2: is the first rain after a dry stretch worse than normal rain?

    Compares collisions on first-rain days against every other rainy day
    with a two sample t-test. Dry days are also averaged for context.

    Args:
        weather_collisions: joined daily weather and collision table.

    Returns:
        A dictionary with the group sizes, averages, the difference
        between them, and the t-test statistic and p-value.
    """
    df = weather_collisions.dropna(subset=['PRCP'])
    rainy_days = df[df['is_rainy']]
    first_rain = rainy_days[rainy_days['is_first_rain']]['crashes']
    other_rain = rainy_days[~rainy_days['is_first_rain']]['crashes']
    t_stat, p_value = stats.ttest_ind(first_rain, other_rain)
    return {
        'n_first_rain': len(first_rain),
        'n_other_rain': len(other_rain),
        'mean_first_rain': first_rain.mean(),
        'mean_other_rain': other_rain.mean(),
        'mean_dry': df[~df['is_rainy']]['crashes'].mean(),
        'difference': first_rain.mean() - other_rain.mean(),
        't_stat': t_stat,
        'p_value': p_value,
    }


def correlate_rain_and_ridership(weather_bikes):
    """
    RQ3: how much does rain cut down bike ridership?

    Runs a Pearson correlation between daily rain and daily bridge
    crossings, and also reports average ridership on rainy vs dry days so
    the correlation has a plain-English size attached to it.

    Args:
        weather_bikes: joined daily weather and ridership table.

    Returns:
        A dictionary with the correlation, its p-value, average riders in
        each group, and the percent drop on rainy days.
    """
    df = weather_bikes.dropna(subset=['PRCP', 'riders'])
    correlation, p_value = stats.pearsonr(df['PRCP'], df['riders'])
    rainy = df[df['is_rainy']]['riders']
    dry = df[~df['is_rainy']]['riders']
    return {
        'n_days': len(df),
        'correlation': correlation,
        'p_value': p_value,
        'mean_riders_rainy': rainy.mean(),
        'mean_riders_dry': dry.mean(),
        'pct_ridership_drop': 100 * (dry.mean() - rainy.mean()) / dry.mean(),
    }


def compare_bike_car_rain_risk(full):
    """
    RQ4: does rain hit cyclists harder than drivers?

    Builds a two by two table of bike-involved vs car-only collisions on
    rainy vs dry days and runs a chi-square test on it. Raw counts alone
    are misleading, because fewer people ride in the rain, so bike
    collisions are also divided by bridge crossings to get a rate per
    thousand riders.

    Args:
        full: joined table with weather, collisions, and ridership.

    Returns:
        A dictionary with the table, the chi-square results, the smallest
        expected count, and the crash rates once ridership is taken into
        account.
    """
    df = full.dropna(subset=['PRCP', 'riders'])
    totals = df.groupby('is_rainy')[
        ['bike_crashes', 'car_only_crashes', 'riders']].sum()
    counts = totals[['bike_crashes', 'car_only_crashes']]
    chi2, p_value, dof, expected = stats.chi2_contingency(counts)

    rate_rainy = (CROSSINGS_PER_UNIT * totals.loc[True, 'bike_crashes']
                  / totals.loc[True, 'riders'])
    rate_dry = (CROSSINGS_PER_UNIT * totals.loc[False, 'bike_crashes']
                / totals.loc[False, 'riders'])
    return {
        'contingency_table': counts,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'min_expected_count': expected.min(),
        'bike_share_rainy': (counts.loc[True, 'bike_crashes']
                             / counts.loc[True].sum()),
        'bike_share_dry': (counts.loc[False, 'bike_crashes']
                           / counts.loc[False].sum()),
        'bike_crashes_per_1k_rainy': rate_rainy,
        'bike_crashes_per_1k_dry': rate_dry,
        'risk_ratio': rate_rainy / rate_dry,
        'car_crashes_per_day_rainy': df[df['is_rainy']][
            'car_only_crashes'].mean(),
        'car_crashes_per_day_dry': df[~df['is_rainy']][
            'car_only_crashes'].mean(),
    }


def plot_rainy_vs_dry(weather_collisions):
    """
    Figure for RQ1: average collisions on rainy vs dry days.

    The bars carry 95% confidence intervals so the gap can be compared
    against how uncertain each average is.

    Args:
        weather_collisions: joined daily weather and collision table.
    """
    df = weather_collisions.dropna(subset=['PRCP'])
    dry = df[~df['is_rainy']]['crashes']
    rainy = df[df['is_rainy']]['crashes']
    means = [dry.mean(), rainy.mean()]
    errors = [confidence_interval(dry), confidence_interval(rainy)]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Dry days', 'Rainy days'], means, yerr=errors, capsize=6,
           width=0.55, color=[DRY_COLOR, RAINY_COLOR])
    ax.set_ylim(0, max(means) * 1.2)
    for i in range(len(means)):
        ax.text(i, means[i] + errors[i] + 0.6, str(round(means[i], 1)),
                ha='center')
    ax.set_ylabel('Average collisions per day')
    ax.set_title('Average Daily Collisions: Rainy vs Dry Days')
    ax.grid(axis='y', color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.savefig(PLOT_DIR + '/rq1_rainy_vs_dry.png', bbox_inches='tight',
                dpi=200)
    plt.close(fig)


def plot_first_rain(weather_collisions):
    """
    Figure for RQ2: dry days, ordinary rainy days, and first-rain days.

    Args:
        weather_collisions: joined daily weather and collision table.
    """
    df = weather_collisions.dropna(subset=['PRCP'])
    rainy_days = df[df['is_rainy']]
    groups = [df[~df['is_rainy']]['crashes'],
              rainy_days[~rainy_days['is_first_rain']]['crashes'],
              rainy_days[rainy_days['is_first_rain']]['crashes']]
    labels = ['Dry', 'Rainy\n(not first)', 'First rain after\n7+ dry days']
    means = [group.mean() for group in groups]
    errors = [confidence_interval(group) for group in groups]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=errors, capsize=6, width=0.55,
           color=[DRY_COLOR, RAINY_COLOR, FIRST_RAIN_COLOR])
    ax.set_ylim(0, max(means) * 1.3)
    for i in range(len(means)):
        ax.text(i, means[i] + errors[i] + 1.0,
                str(round(means[i], 1)) + '\n(n=' + str(len(groups[i])) + ')',
                ha='center', fontsize=9)
    ax.set_ylabel('Average collisions per day')
    ax.set_title('Collisions on First-Rain Days vs Other Days')
    ax.grid(axis='y', color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.savefig(PLOT_DIR + '/rq2_first_rain.png', bbox_inches='tight',
                dpi=200)
    plt.close(fig)


def plot_rain_vs_ridership(weather_bikes):
    """
    Figure for RQ3: daily rain against daily bridge crossings.

    Args:
        weather_bikes: joined daily weather and ridership table.
    """
    df = weather_bikes.dropna(subset=['PRCP', 'riders'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['PRCP'], df['riders'], s=5, alpha=0.25, color=DRY_COLOR)
    ax.set_xlabel('Daily precipitation (inches)')
    ax.set_ylabel('Bridge crossings per day')
    ax.set_title('Precipitation vs Daily Bike Ridership')
    ax.grid(color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.savefig(PLOT_DIR + '/rq3_rain_vs_ridership.png',
                bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_ridership_by_rain_band(weather_bikes):
    """
    Figure for RQ3: average ridership grouped by how hard it rained.

    The scatter plot shows the overall shape but the points overlap
    heavily, so this groups days into rain amounts and shows the average
    for each group, which is easier to read a size off of.

    Args:
        weather_bikes: joined daily weather and ridership table.
    """
    df = weather_bikes.dropna(subset=['PRCP', 'riders'])
    precip = df['PRCP']
    groups = [df[precip == 0]['riders'],
              df[(precip > 0) & (precip <= 0.1)]['riders'],
              df[(precip > 0.1) & (precip <= 0.5)]['riders'],
              df[precip > 0.5]['riders']]
    labels = ['No rain', 'Up to 0.1"', '0.1-0.5"', 'Over 0.5"']
    means = [group.mean() for group in groups]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, width=0.55, color=DRY_COLOR)
    for i in range(len(means)):
        ax.text(i, means[i] + 80, str(round(means[i])), ha='center')
    ax.set_xlabel('Daily precipitation')
    ax.set_ylabel('Average bridge crossings per day')
    ax.set_title('Average Bike Ridership by Amount of Rain')
    ax.grid(axis='y', color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.savefig(PLOT_DIR + '/rq3_ridership_by_band.png',
                bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_bike_risk(results):
    """
    Figure for RQ4: bike collisions per thousand bridge crossings.

    Args:
        results: the dictionary returned by compare_bike_car_rain_risk.
    """
    means = [results['bike_crashes_per_1k_dry'],
             results['bike_crashes_per_1k_rainy']]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Dry days', 'Rainy days'], means, width=0.55,
           color=[DRY_COLOR, RAINY_COLOR])
    ax.set_ylim(0, max(means) * 1.15)
    for i in range(len(means)):
        ax.text(i, means[i] * 1.02, str(round(means[i], 3)), ha='center')
    ax.set_ylabel('Bike collisions per 1,000 crossings')
    ax.set_title('Bike Collision Risk per Rider: Rainy vs Dry Days')
    ax.grid(axis='y', color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.savefig(PLOT_DIR + '/rq4_bike_risk.png', bbox_inches='tight',
                dpi=200)
    plt.close(fig)


def print_results(title, results):
    """
    Print one research question's numbers in a readable block.

    Args:
        title: heading to print above the numbers.
        results: dictionary of names and values to print.
    """
    print()
    print('=' * 55)
    print(title)
    print('=' * 55)
    for name in results:
        value = results[name]
        if isinstance(value, float):
            print('  ' + name.ljust(28) + format(value, ',.5g'))
        elif name != 'contingency_table':
            print('  ' + name.ljust(28) + str(value))


def main():
    """Load the data, answer all four questions, and save the figures."""
    weather_collisions, weather_bikes, full = dp.load_all(
        DATA_DIR + '/NOAA.csv',
        DATA_DIR + '/Fremont_Bridge.csv', FREMONT_COL,
        DATA_DIR + '/Spokane_St.csv', SPOKANE_COL,
        DATA_DIR + '/SDOT.csv')

    rq1 = compare_rainy_dry_crashes(weather_collisions)
    rq2 = compare_first_rain_crashes(weather_collisions)
    rq3 = correlate_rain_and_ridership(weather_bikes)
    rq4 = compare_bike_car_rain_risk(full)

    print_results('RQ1: collisions on rainy vs dry days', rq1)
    print_results('RQ2: first rain after a dry stretch', rq2)
    print_results('RQ3: rain vs bike ridership', rq3)
    print_results('RQ4: bike vs car collisions in rain', rq4)
    print()
    print('  contingency table (rows: is_rainy)')
    print(rq4['contingency_table'].to_string())

    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_rainy_vs_dry(weather_collisions)
    plot_first_rain(weather_collisions)
    plot_rain_vs_ridership(weather_bikes)
    plot_ridership_by_rain_band(weather_bikes)
    plot_bike_risk(rq4)
    print()
    print('Saved figures to ' + PLOT_DIR + '/')


if __name__ == '__main__':
    main()
