# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %run 'carbon alloca_ZSG DEA.py'
from pandas import ExcelWriter

import numpy as np
import os
import pandas as pd
import pickle
import pulp


# # %load_ext nb_black

# +
file_lstm = r"Data/Data_lstm.pickle"


def read_dataframe(path, file_lstm=file_lstm):
    """
    This function read dataFrame from a pickle file.
    THe pickle file stores predicted macroeconomic data of China from 2000-2030

    read_dataframe(path, file_lstm=file_lstm)
        Return DataFrame contains labor, capital, energy consumption, GDP, and CO2 emission data of 30 provinces in 2000-2030

        Parameters
        ----------
        path: file path
        file_lstm: name of the pickle file, default is (file_lstm = r"Data_lstm.pickle")
    """
    # os.chdir(path)
    with open(file_lstm, "rb") as file:
        data_df = pickle.load(file)
    return data_df


# -

def read_data(data, column_name, year):
    """
    This function find desired data through screening

    read_data(data, column_name, year)
        Return DataFrame contains desired data content

        Paramters
        ---------
        data: DataFrame, original DataFrame contains all data
        column_name: list/string, desired columns from data
        year: int, desired year
    """
    data_col = data.loc[:, column_name]
    data_col_year = data_col.loc[data_col.index.get_level_values(1) == year]
    return data_col_year


def calc_eff(year, data, weight, disp="weak disposability"):
    data_in = read_data(data, column_in, year)
    data_out = read_data(data, column_out, year)
    data_undout = read_data(data, column_undout, year)
    names = pd.DataFrame([i for i, _ in read_data(data, column_undout, year).index])
    solve = DEAProblem(data_in, data_out, data_undout, weight, disp=disp).solve()
    status = pd.DataFrame.from_dict(solve[0], orient="index", columns=["status"])
    efficiency = pd.DataFrame.from_dict(
        solve[1], orient="index", columns=["efficiency"]
    )
    weight = pd.DataFrame.from_dict(solve[2], orient="index")
    results = pd.concat([names, status, efficiency, weight], axis=1)
    return results


path = r"Data"


def read_zsg_data(year, path=path):
    path = os.path.join(path, "DEA_results")
    file_name_list = os.listdir()
    file = [file for file in file_name_list if str(year) in file][0]
    with open(file, "rb") as zsg_data:
        data_df = pickle.load(zsg_data).loc[:, [0, "hri_score"]]
    return data_df


def zsg_carbon_emission(year, original_data):
    zsg_eff = read_zsg_data(year)
    data_undout = read_data(original_data, column_undout, year)
    zsg_carbon = pd.DataFrame(
        np.array(data_undout.iloc[:, -1]) * np.array(zsg_eff.loc[:, "hri_score"]),
        columns=["zsg_CO2"],
    )
    zsg_carbon = pd.concat([zsg_eff, zsg_carbon], axis=1)
    return zsg_carbon


# +
data = read_dataframe(os.path.join(path))

column_in = ["Population", "Fixed asset", "Energy consumption"]
column_out = ["GDP"]
column_undout = ["CO2 emisson"]
weight = [0, 0, 0, 1 / 2, 1 / 2]


# -

def concat_zsg_data(year, DMU_name, original_data=data):
    """
    This function concatnate
    """
    zsg_carbon = zsg_carbon_emission(year, original_data)
    zsg_CO2_DMU = zsg_carbon[zsg_carbon.loc[:, 0] == DMU_name].iloc[0, -1]
    data_year = original_data[original_data.index.get_level_values(1) == year]
    zsg_inout = original_data.loc[[(DMU_name, year)], :]
    zsg_inout.iloc[:, -1] = zsg_CO2_DMU
    ddf_data = pd.concat([data_year, zsg_inout], axis=0)
    return ddf_data


def calc_econ_benefit(path, year, DMU_name, weight=weight, disp="weak disposability"):
    """
    This function calculate economic welfare change due to carbon emission right allocation

    calc_econ_benefit(path, year,DMU_name,weight=weight, disp = "weak disposability")
        Return the economic welfare change of a DMU in year due to carbon emission right allocation (DataFrame)

        Paramters
        ---------
        path: the path that store the lstm prediction results on macroeconomic factors
        year: stands for the year to calculate carbon emission right allocation
        DMU_name: represents the DMU to calculate economic welfare change
        weight: weight vector in DDF calculation
        disp: assumption adopted in DEA model
    """

    zsg_ddf_data = concat_zsg_data(year, DMU_name)
    zsg_results = calc_eff(year, zsg_ddf_data, weight, disp=disp)

    scal_fac_b0 = zsg_results[zsg_results.loc[:, 0] == DMU_name][
        "scalingFactor_b_0"
    ].iloc[0]
    scal_fac_y0 = zsg_results[zsg_results.loc[:, 0] == DMU_name][
        "scalingFactor_y_0"
    ].iloc[0]
    scal_fac_b1 = zsg_results[zsg_results.loc[:, 0] == DMU_name][
        "scalingFactor_b_0"
    ].iloc[1]
    scal_fac_y1 = zsg_results[zsg_results.loc[:, 0] == DMU_name][
        "scalingFactor_y_0"
    ].iloc[1]

    scal_fac_y0_potential = ((1 - scal_fac_b0) / 1 - (1 - scal_fac_b1) / 1) / (
        (1 + scal_fac_y0) * (1 - scal_fac_b0)
    )
    # the expression of economic welfare change coefficient calculate economic welfare change due to emission right change
    # the expression assumes carbon productivity of additional carbon emissions keeps constant
    # the first item (1 - scal_fac_b0) / 1 - (1 - scal_fac_b1) / 1) expresses the change in carbon emission efficiency
    # the item (1 - scal_fac_b0) / 1 expresses original carbon emission efficiency
    # the item (1 - scal_fac_b1) / 1 expresses carbon emission efficiency after emission right allocation
    # the second term 1/((1 + scal_fac_y0) * (1 - scal_fac_b0)) express carbon productivity of original DMU
    # the item 1/(1 + scal_fac_y0) expresses economic efficiency
    # the item 1/(1 - scal_fac_b0) expresses carbon emission efficiency

    data = read_dataframe(os.path.join(path))
    data_out = read_data(data, column_out, year)
    gdp_DMU = data_out[data_out.index.get_level_values(0) == DMU_name]
    eco_ben = scal_fac_y0_potential * gdp_DMU
    return eco_ben


def yearly_econ_benefit(year, path=path):
    data = read_dataframe(os.path.join(path))
    data_undout = read_data(data, column_undout, year)
    data_out = read_data(data, column_out, year)
    ls = []
    for i, _ in data_out.index:
        ls.append(calc_econ_benefit(path, year, i))

    # test = pd.DataFrame(ls)
    econ_benefit_this_year = pd.concat(ls)
    zsg_eff = read_zsg_data(year)
    eco_benefit_df = pd.concat(
        [
            data_out,
            econ_benefit_this_year.rename(columns={"GDP": "economic_impact"}),
            pd.DataFrame(
                np.array(econ_benefit_this_year.iloc[:, -1])
                / (
                    np.array(data_undout.iloc[:, -1])
                    * (np.array(zsg_eff.loc[:, "hri_score"]) - 1)
                ),
                index=data_out.index,
            ).rename(columns={0: "carbon_price"}),
        ],
        axis=1,
    )
    return eco_benefit_df


def export_excel(year_range, path=path):

    # os.chdir(path)
    with ExcelWriter("economic_impct.xlsx") as writer:
        for year in year_range:
            eco_ben_df = yearly_econ_benefit(year, path=path)
            eco_ben_df.to_excel(writer, sheet_name=str(year))
        writer.save()


def main():
    year_range = range(2000, 2031)
    export_excel(year_range)


main()

yearly_econ_benefit(2001, path=path)
