"""
Rain City Roads, testing program
Elsa Chao, CSE 163, Summer 2026

Tests data_processing.py and analysis.py against small CSV files and small
tables whose right answers were worked out by hand. Every check is an
assert statement, so a wrong answer stops the program instead of printing
something that looks fine. Run this file directly to run all the tests.
"""
import pandas as pd

import data_processing as dp
import analysis

TOLERANCE = 0.000001


def assert_close(actual, expected, label):
    """
    Check a number matches the value worked out by hand.

    Args:
        actual: the value the code produced.
        expected: the value it should be.
        label: name to show if the check fails.
    """
    assert abs(actual - expected) < TOLERANCE, \
        label + ': got ' + str(actual) + ', expected ' + str(expected)


def test_load_bridge():
    """
    Hourly counts should add up into the right daily totals.

    The test file has three hourly rows over two days, 10 + 20 on the
    first day and 5 on the second, so the totals have to be 30 and 5.
    """
    daily = dp.load_bridge('data/test_bridge.csv', 'Total')
    assert list(daily.columns) == ['date', 'riders']
    assert len(daily) == 2
    assert daily['riders'].tolist() == [30, 5]


def test_load_bridge_strips_commas():
    """
    Counts stored as text with commas have to become real numbers.

    The second test file writes 1,000 with a comma. If it stayed as text
    the daily total would come out wrong, so this checks 1,000 + 500
    comes back as 1500.
    """
    daily = dp.load_bridge('data/test_bridge_b.csv',
                           'Spokane St. Bridge Total')
    assert daily['riders'].tolist() == [1500, 7]


def test_combine_bridges():
    """
    Combining the counters should keep only days both bridges reported.

    The two test files only share 2020-01-01, so there should be one row
    left and the total should be 30 + 1500.
    """
    fremont = dp.load_bridge('data/test_bridge.csv', 'Total')
    spokane = dp.load_bridge('data/test_bridge_b.csv',
                             'Spokane St. Bridge Total')
    bikes = dp.combine_bridges(fremont, spokane)
    assert len(bikes) == 1
    assert bikes['riders'].tolist() == [1530]


def test_load_collisions():
    """
    Collisions should be counted per day and split into two groups.

    The test file has one collision on 01-02 with a cyclist, three on
    01-10 where one had a cyclist, and two on 01-11 with none. Bike and
    car-only counts also have to add back up to the daily total, since
    RQ4 needs groups that don't overlap.
    """
    daily = dp.load_collisions('data/test_collisions.csv')
    assert len(daily) == 3
    assert daily['crashes'].tolist() == [1, 3, 2]
    assert daily['bike_crashes'].tolist() == [1, 1, 0]
    assert daily['car_only_crashes'].tolist() == [0, 2, 2]
    totals = daily['bike_crashes'] + daily['car_only_crashes']
    assert totals.tolist() == daily['crashes'].tolist()


def test_add_rain_flags():
    """
    A first-rain day needs a full dry stretch right before it.

    In the test file 01-10 is rainy after eight dry days, so it counts.
    01-01 is rainy with nothing recorded before it, 01-11 is rainy but
    comes after a rainy day, and 01-17 is rainy after only five dry
    days, so none of those count.
    """
    weather = dp.add_rain_flags(dp.load_weather('data/test_weather.csv'))
    assert len(weather) == 20
    rainy = weather[weather['is_rainy']]['date'].dt.day.tolist()
    assert rainy == [1, 10, 11, 17]
    first = weather[weather['is_first_rain']]['date'].dt.day.tolist()
    assert first == [10]


def test_add_rain_flags_ignores_missing_days():
    """
    A gap in the weather record is not a stretch of dry weather.

    The gap test file has eight dry rows in a row before a rainy day,
    but five calendar days in the middle are missing completely. Counting
    rows instead of dates would wrongly call that a first rain, so the
    right answer is that no day counts.
    """
    weather = dp.add_rain_flags(
        dp.load_weather('data/test_weather_gap.csv'))
    assert not weather['is_first_rain'].any()


def test_build_tables():
    """
    Joining should keep only the days every dataset covers.

    Weather covers 20 days, collisions cover three of them, and the two
    bridges only agree on 01-01. So weather + collisions has three rows,
    weather + bikes has one, and all four together has none, because
    01-01 has no collisions in the test data.
    """
    weather = dp.add_rain_flags(dp.load_weather('data/test_weather.csv'))
    bikes = dp.combine_bridges(
        dp.load_bridge('data/test_bridge.csv', 'Total'),
        dp.load_bridge('data/test_bridge_b.csv',
                       'Spokane St. Bridge Total'))
    collisions = dp.load_collisions('data/test_collisions.csv')
    weather_collisions, weather_bikes, full = dp.build_tables(
        weather, bikes, collisions)
    assert len(weather_collisions) == 3
    assert len(weather_bikes) == 1
    assert len(full) == 0


def make_fake_crashes():
    """
    Build a small collision table with a gap I already know the size of.

    Each of four months gets five rainy days averaging 20 crashes and
    five dry days averaging 10, so every average is known ahead of time.

    Returns:
        A DataFrame shaped like the weather + collisions table.
    """
    rainy_counts = [18, 19, 20, 21, 22]
    dry_counts = [8, 9, 10, 11, 12]
    rows = []
    for month in range(1, 5):
        for i in range(len(rainy_counts)):
            rows.append({'date': pd.Timestamp(2021, month, i + 1),
                         'PRCP': 0.2, 'is_rainy': True,
                         'is_first_rain': False,
                         'crashes': rainy_counts[i]})
        for i in range(len(dry_counts)):
            rows.append({'date': pd.Timestamp(2021, month, i + 10),
                         'PRCP': 0.0, 'is_rainy': False,
                         'is_first_rain': False,
                         'crashes': dry_counts[i]})
    return pd.DataFrame(rows)


def test_compare_rainy_dry_crashes():
    """
    The RQ1 numbers have to match the table I built by hand.

    Rainy days average 20 and dry days average 10, so the increase is
    100%, and a gap that size is way outside random noise, so the
    p-value has to be tiny.
    """
    results = analysis.compare_rainy_dry_crashes(make_fake_crashes())
    assert results['n_rainy'] == 20
    assert results['n_dry'] == 20
    assert_close(results['mean_rainy'], 20.0, 'mean_rainy')
    assert_close(results['mean_dry'], 10.0, 'mean_dry')
    assert_close(results['pct_increase'], 100.0, 'pct_increase')
    assert results['p_value'] < 0.001


def test_compare_rainy_dry_no_difference():
    """
    Two identical groups should not come out significant.

    This is the other half of the check. If the code found a difference
    here, the real RQ1 answer couldn't be trusted either, so rainy days
    are given the exact same numbers as dry days and the p-value has to
    be large.
    """
    df = make_fake_crashes()
    df.loc[df['is_rainy'], 'crashes'] = [8, 9, 10, 11, 12] * 4
    results = analysis.compare_rainy_dry_crashes(df)
    assert_close(results['mean_rainy'], results['mean_dry'], 'means match')
    assert results['p_value'] > 0.9


def test_compare_first_rain_crashes():
    """
    First-rain days should be pulled out of the rainy days correctly.

    Two of the rainy days are marked as first rain and given 39 and 41
    crashes, so the first-rain average is 40 and the count is 2.
    """
    df = make_fake_crashes()
    df.loc[[0, 10], 'is_first_rain'] = True
    df.loc[[0, 10], 'crashes'] = [39, 41]
    results = analysis.compare_first_rain_crashes(df)
    assert results['n_first_rain'] == 2
    assert results['n_other_rain'] == 18
    assert_close(results['mean_first_rain'], 40.0, 'mean_first_rain')
    assert_close(results['mean_dry'], 10.0, 'mean_dry')


def test_compare_bike_car_rain_risk():
    """
    Dividing by ridership has to use the matching group's riders.

    The table gives rainy days 10 bike collisions over 2,000 crossings
    and dry days 8 over 8,000, so the rates are 5.0 and 1.0 per thousand
    and the ratio between them is exactly 5. It also checks the expected
    counts clear the minimum of 5 that a chi-square test needs.
    """
    rows = [
        {'date': pd.Timestamp(2021, 1, 1), 'PRCP': 0.3, 'is_rainy': True,
         'bike_crashes': 5, 'car_only_crashes': 45, 'riders': 1000},
        {'date': pd.Timestamp(2021, 1, 2), 'PRCP': 0.3, 'is_rainy': True,
         'bike_crashes': 5, 'car_only_crashes': 45, 'riders': 1000},
        {'date': pd.Timestamp(2021, 1, 3), 'PRCP': 0.0, 'is_rainy': False,
         'bike_crashes': 4, 'car_only_crashes': 32, 'riders': 4000},
        {'date': pd.Timestamp(2021, 1, 4), 'PRCP': 0.0, 'is_rainy': False,
         'bike_crashes': 4, 'car_only_crashes': 32, 'riders': 4000},
    ]
    results = analysis.compare_bike_car_rain_risk(pd.DataFrame(rows))
    assert_close(results['bike_crashes_per_1k_rainy'], 5.0, 'rainy rate')
    assert_close(results['bike_crashes_per_1k_dry'], 1.0, 'dry rate')
    assert_close(results['risk_ratio'], 5.0, 'risk ratio')
    assert_close(results['car_crashes_per_day_rainy'], 45.0, 'car rainy')
    assert_close(results['car_crashes_per_day_dry'], 32.0, 'car dry')
    assert results['min_expected_count'] >= 5
    assert results['dof'] == 1


def test_confidence_interval():
    """
    More days should give a tighter confidence interval.

    Both samples have the same spread but one has twenty times as many
    values, so its interval has to be smaller. That is what makes the
    error bars in the RQ1 figure mean something.
    """
    small = pd.Series([8, 9, 10, 11, 12])
    large = pd.Series([8, 9, 10, 11, 12] * 20)
    assert (analysis.confidence_interval(large)
            < analysis.confidence_interval(small))


def main():
    """Run every test and say so if they all pass."""
    tests = [
        test_load_bridge,
        test_load_bridge_strips_commas,
        test_combine_bridges,
        test_load_collisions,
        test_add_rain_flags,
        test_add_rain_flags_ignores_missing_days,
        test_build_tables,
        test_compare_rainy_dry_crashes,
        test_compare_rainy_dry_no_difference,
        test_compare_first_rain_crashes,
        test_compare_bike_car_rain_risk,
        test_confidence_interval,
    ]
    for test in tests:
        test()
        print('passed: ' + test.__name__)
    print()
    print('all ' + str(len(tests)) + ' tests passed')


if __name__ == '__main__':
    main()
