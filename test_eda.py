"""Tests for eda.py using a small hand-checkable dataset."""
from eda import load_bridge


def test_load_bridge():
    """Daily sums should match hand-computed totals."""
    df = load_bridge('data/test_bridge.csv', 'Total')
    assert len(df) == 2
    assert df['riders'].tolist() == [30, 5]
    print('all tests passed')


if __name__ == '__main__':
    test_load_bridge()