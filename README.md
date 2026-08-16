# Rain City Roads

This is my CSE 163 final project for Summer 2026. I looked at whether Seattle
weather actually changes how many crashes happen and how many people bike.
The writeup is on Gradescope, this repo is just the code.

There are four questions I tried to answer. Whether collisions go up when it
rains, whether the first rain after a dry spell is really the worst one like
everyone says, how much rain cuts down bike ridership, and whether cyclists
get hit harder by rain than drivers once you account for the fact that way
fewer people are out riding.

## Setup

You need Python 3.9 or newer and three libraries:

```
pip install pandas scipy matplotlib
```

flake8 is optional, only if you want to check the style.

## Data

The data files are way too big to put in here so you have to download them
yourself. Make a folder called `data` in the project root and put all four
files in it. The names have to match exactly or the code won't find them.

Three of them come from data.seattle.gov. Search for "SDOT Collisions All
Years" and save it as `data/SDOT.csv`, then the Fremont Bridge Combined
Bicycle and Scooter Counter as `data/Fremont_Bridge.csv`, and the Spokane St.
Bridge one as `data/Spokane_St.csv`. All three download as CSV.

The weather is from NOAA at ncei.noaa.gov/cdo-web. Ask for daily summaries
from station USW00024233, which is SeaTac airport, and request PRCP, TMAX and
TMIN in standard units. Save that as `data/NOAA.csv`.

Don't worry if your downloads have extra columns, mine did too. The code only
uses INCDATE and PEDCYLCOUNT from the collisions file and DATE, PRCP, TMAX and
TMIN from the weather one.

One thing, there are already some tiny files in the data folder that start
with `test_`. Those are fake data I typed up by hand for my tests, so leave
them alone.

## What each file is

`data_processing.py` does all the loading and cleaning. It reads each dataset,
adds everything up into daily totals, marks which days were rainy and which
ones were the first rain after a dry stretch, and joins everything together on
the date. You can't run this one on its own, the other two files import it.

`analysis.py` is the one you actually run. It answers all four questions and
saves the figures I used in the report.

`test_project.py` is my testing file. It has 12 tests and they all use assert
statements.

There's also `eda.py` and `test_eda.py` from Part 2. Those are replaced by the
files above but I left them in.

## Running it

Once your four data files are in place, from the project root just run:

```
python analysis.py
```

Give it a few seconds, the collisions file has 261,009 rows in it so it takes
a moment to load. It prints out the numbers for all four questions and saves
five figures into a `plots` folder. If you don't have a plots folder it makes
one for you.

For the tests:

```
python test_project.py
```

Those only use the little fake `test_` files, so you can run them without
downloading anything. You should get 12 lines saying passed and then "all 12
tests passed" at the bottom. If something's wrong you'll get an AssertionError
telling you which number didn't match.

## Some things that tripped me up

The two bridge files use completely different date formats, which took me
forever to figure out. One of them also writes its counts with commas in them
like "1,234" so pandas reads them as text instead of numbers, and my totals
were silently wrong until I caught it. Both of those are handled in
`data_processing.py` now and there are tests for them.

Also the Spokane St counter doesn't start until January 2014 even though the
Fremont one goes back to 2012, so anything involving bikes only covers 2014
to 2026. And if only one bridge reported on a day I drop that day, since
otherwise one bridge would be pretending to be two.