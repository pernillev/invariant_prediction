# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 08:05:00 2021

@author: Pernille
"""

import simulate_functions
from BHIP import bhip_fit


def main(seeds):
    list_of_df = list()
    for i in range(1):
        df = simulate_functions.simulate([4], [250], 3, 2, 2, seeds[i])
        list_of_df.append(df[0])
    # %%
    list_of_fit = list()
    for df in list_of_df:
        fit = bhip_fit(df)
        list_of_fit.append(fit)
    return list_of_fit


if __name__ == '__main__':
    seeds = [321, 142, 432, 244, 562, 623, 145, 874, 765, 654]
    fits = main(seeds)
