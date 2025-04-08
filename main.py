import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, find_peaks
from sklearn.neighbors import KernelDensity

from config import *


def draw_candle_chart(sample_df: pd, lines: np, color_1: str, color_2: str, region=None):
    f = plt.figure()
    f.suptitle('Candle chart')
    f.set_figwidth(15)
    # define width of candlestick elements
    width = .4
    width2 = .05
    # define up and down prices
    up = sample_df[sample_df.close >= sample_df.open]
    down = sample_df[sample_df.close < sample_df.open]
    # plot up prices
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=color_1)
    plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=color_1)
    plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=color_1)
    # plot down prices
    plt.bar(down.index, down.close - down.open, width, bottom=down.open, color=color_2)
    plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color=color_2)
    plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color=color_2)
    # rotate x-axis tick labels
    plt.xticks(rotation=45, ha='right')

    for x in lines:
        plt.hlines(x, xmin=sample_df.index[0], xmax=sample_df.index[-1])
        if region is not None:
            plt.fill_between(sample_df.index, x - x * region, x + x * region, alpha=0.4)


def draw_line_chart(sample_df: pd, lines: np, region=None, mavg=None):
    f = plt.figure()
    f.suptitle('Line chart')
    f.set_figwidth(15)
    if mavg is not None:
        mavg_df = sample_df[['open', 'high', 'low', 'close']].rolling(window=mavg).mean()
        plt.plot(mavg_df.index, mavg_df.close)
    else:
        plt.plot(sample_df.index, sample_df.close)
    # rotate x-axis tick labels
    plt.xticks(rotation=45, ha='right')

    for x in lines:
        plt.hlines(x, xmin=sample_df.index[0], xmax=sample_df.index[-1])
        if region is not None:
            plt.fill_between(sample_df.index, x - x * region, x + x * region, alpha=0.4)


def draw_turning_points(sample_original: np, extrema_prices: np):
    f = plt.figure()
    f.suptitle('Turning points')
    f.set_figwidth(15)
    plt.hlines(1, sample_original.min(), sample_original.max())
    plt.eventplot(extrema_prices, orientation='horizontal', colors='b')


def main():
    data = pd.read_csv(csv_file, usecols=[1, 2, 3, 4, 5]
                       # , names= ['data', 'open', 'high', 'low', 'close']
                       )
    num_peaks = -999
    sample_df = data.iloc[slice_]
    sample = data.iloc[slice_][['close']].to_numpy().flatten()
    sample_original = sample.copy()
    maxima = argrelextrema(sample, np.greater)
    minima = argrelextrema(sample, np.less)
    extrema = np.concatenate((maxima, minima), axis=1)[0]
    extrema_prices = np.concatenate((sample[maxima], sample[minima]))
    interval = extrema_prices[0] / 10000
    bandwidth = interval
    while num_peaks < peaks_range[0] or num_peaks > peaks_range[1]:
        initial_price = extrema_prices[0]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(extrema_prices.reshape(-1, 1))
        a, b = min(extrema_prices), max(extrema_prices)
        price_range = np.linspace(a, b, 1000).reshape(-1, 1)
        pdf = np.exp(kde.score_samples(price_range))
        peaks = find_peaks(pdf)[0]
        num_peaks = len(peaks)
        bandwidth += interval
        if bandwidth > 100 * interval:
            print("Failed to converge, stop")
            break

    draw_turning_points(sample_original, extrema_prices)
    draw_line_chart(sample_df, price_range[peaks], region=0.0001, mavg=3)
    draw_candle_chart(sample_df, price_range[peaks], color_1, color_2, region=0.0001)

    plt.show()


if __name__ == '__main__':
    main()
