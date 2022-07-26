import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from decouple import config as envs
from sqlalchemy import create_engine
from eaglegaze_common import entsoe_configs, logger
from eaglegaze_common.common_attr import Attributes as at
from eaglegaze_common.common_utils import insert_into_table, substract_time_shift, start_end_microservice_time, \
    reduce_memory_usage
import warnings
from multiprocessing import Pool, Process


warnings.filterwarnings("ignore")
logger = logger.get_logger(__name__, at.LogAttributes.log_file)


class Margin_cost():
    def __connection(self):
        engine = create_engine(envs('ALCHEMY_CONNECTION', cast=str))
        con = engine.raw_connection()
        cur = con.cursor()
        return cur

    def get_tables(self, schema, table_name):
        cur = self.__connection()
        cur.execute(
            f"select * from {schema}.{table_name}")
        table_df = pd.DataFrame(cur.fetchall())
        table_df.columns = [d[0] for d in cur.description]
        logger.info(f'got table {table_name}')
        return table_df

    def get_tables_iso(self, schema, table_name):
        engine = create_engine(envs('ALCHEMY_CONNECTION', cast=str))
        con = engine.raw_connection()
        cur = con.cursor()
        cur.execute(
            f"select * from {schema}.{table_name}")
        table_df = pd.DataFrame(cur.fetchall())
        table_df.columns = [d[0] for d in cur.description]
        return table_df

    def get_powerunit_country(self, generation_type):
        cur = self.__connection()
        cur.execute(
            f"select t.unit_id, effectiveness, for_model,  generation_id, iso_code, t3.id from im.power_powerunit_info t left join im.power_powerstation_info t1 on station_id=id "
            f"left join im.power_co2_info t2 on t.eic_code=t2.powerunit_eic_code left join bi.countries t3 on country = t3.id "
            f"where generation_id={generation_type}")
        power_df = pd.DataFrame(cur.fetchall())
        power_df.columns = [d[0] for d in cur.description]
        power_df = power_df[['unit_id', 'effectiveness', 'for_model', 'generation_id', 'iso_code', 'id']]
        power_df = power_df.rename(columns={'generation_id': 'generation_type_id'})
        power_df['for_model'].fillna(0, inplace=True)
        power_df['effectiveness'].fillna(0.3, inplace=True)
        power_df = reduce_memory_usage(power_df)
        logger.info('got power data')
        return power_df

    # def get_powerunits_country(self):
    #     p_df = self.get_powerunit_country(7)
    #     powerunit_data = self.get_tables_iso('im', 'power_powerunit_info')
    #     powerstation_df = self.get_tables('im', 'power_powerstation_info')[['id', 'station', 'country']]
    #     powerstation_df = powerstation_df.rename(columns={'id': 'station_id', 'station': 'powerunit'})
    #     powercountry_df = pd.merge(powerstation_df, powerunit_data, how='outer', on='station_id', indicator=True)
    #     powercountry_df = powercountry_df[['unit_id', 'country', 'eic_code', 'generation_id', 'effectiveness']]
    #     # powerunit_info = self.get_tables('bi', 'power_unit_info_entsoe')
    #     powerunit_co2 = self.get_tables_iso('im', 'power_co2_info')
    #     countries = self.get_tables('bi', 'countries')
    #     countries = countries.rename(columns={'id': 'country'})
    #     powerunit_co2 = powerunit_co2.rename(columns={'powerunit_eic_code': 'eic_code'})
    #     powerunit_info_co2 = pd.merge(powercountry_df, powerunit_co2, how='outer', on='eic_code', indicator=True)
    #     # powerunit_info_co2 = powerunit_info[powerunit_info['_merge'] == 'both']
    #     powerunit_info_co2 = powerunit_info_co2.query("_merge != 'right_only'")[
    #         ['unit_id', 'eic_code', 'country', 'generation_id', 'effectiveness', 'for_model']]
    #     powerunit_info_co2 = powerunit_info_co2.drop_duplicates(subset='unit_id')
    #     # power_df = pd.merge(powerunit_data[['eic_code', 'effectiveness', 'unit_id']], powerunit_info_co2, how='outer', on='eic_code', indicator=True)
    #     power_df = pd.merge(powerunit_info_co2, countries[['country', 'iso_code']], how='outer', on='country',
    #                         indicator=True)
    #     power_df = power_df[power_df['_merge'] == 'both']
    #     power_df = power_df[['unit_id', 'effectiveness', 'for_model', 'generation_id', 'iso_code', 'country']]
    #     power_df = power_df.rename(columns={'generation_id': 'generation_type_id', 'country': 'id'})
    #     power_df['for_model'].fillna(0, inplace=True)
    #     power_df['effectiveness'].fillna(0.3, inplace=True)
    #     power_df = power_df.drop_duplicates(subset='unit_id')
    #     power_df = reduce_memory_usage(power_df)
    #     logger.info('got power data')
    #     return power_df

    def get_first_last_forecast_date(self):
        # получаем дату начала и конца моделирования из таблицы im.im_iteration
        dates_df = self.get_tables('im', 'im_iteration order by iteration_id')[
            ['start_forecast_utc', 'end_forecast_utc']]
        first_forecast_date = pd.to_datetime(dates_df['start_forecast_utc'].values[-1])
        last_forecast_date = pd.to_datetime(dates_df['end_forecast_utc'].values[-1])
        logger.info(f'get first_forecast_date, last_forecast_date: {first_forecast_date, last_forecast_date}')
        return first_forecast_date, last_forecast_date



    def get_the_furthest_year_of_product(self, ticker: int):
        # на случай если в таблице power_installed_country_capacity нет годовой мощности до 2030года
        series_values = self.get_tables('bi', 'series_data order by d_date')
        series_values = series_values.query('series_id == @ticker')
        series_values.sort_values(by='d_date')
        date_df = series_values.query('series_id == @ticker')['d_date']
        ####
        logger.info(f"working with ticker {ticker}")
        ####
        max_year = date_df.iloc[-1].date()
        last_capacity = series_values['value'].iloc[-1]
        max_year = datetime.strptime(f"{max_year}", '%Y-%m-%d').year
        # first_forecast_date, last_forecast_date = self.get_first_last_forecast_date()
        if max_year <= self.last_forecast_date.year + 1:
            d = [datetime.strptime(f"{date}-{month}-01", '%Y-%m-%d') for date in
                 range(max_year, self.last_forecast_date.year + 1) for month in range(1, 13)]
            cap = [last_capacity] * len(d)
            additional_series_values = pd.DataFrame(data={
                'series_id': [series_values['series_id'].values[0]] * 12 * (self.last_forecast_date.year + 1 - max_year),
                'frequency_id': [4] * 12 * (self.last_forecast_date.year + 1 - max_year),
                'd_date': d,
                'value': cap
            })
            series_values = pd.concat([series_values, additional_series_values])
        min_year = date_df.iloc[0].date()
        first_capacity = series_values['value'].iloc[0]
        min_year = datetime.strptime(f"{min_year}", '%Y-%m-%d').year
        if min_year >= self.first_forecast_date.year - 1:
            d = [datetime.strptime(f"{date}-{month}-01", '%Y-%m-%d') for date in
                 range(max_year, self.last_forecast_date.year + 1) for month in range(1, 13)]
            cap = [first_capacity] * len(d)
            additional_series_values = pd.DataFrame(data={
                'series_id': [series_values['series_id'].values[0]] * len(d),
                'frequency_id': [4] * len(d),
                'd_date': d,
                'value': cap
            })
            series_values = pd.concat([series_values, additional_series_values], ignore_index=True)
            series_values.drop_duplicates(subset=['series_id', 'd_date'], keep='first')
            series_values['d_date'] = pd.to_datetime(series_values['d_date'])
            series_values.sort_values(by='d_date', ascending=True)
        return series_values


    def lignite(self, scenario):
        # first_forecast_date, last_forecast_date = self.get_first_last_forecast_date()
        logger.info('running lignite')
        power_df = self.get_powerunit_country(7)
        # maks_units = pd.DataFrame(data={
        #     'unit_id': [2076, 175, 176, 177, 1713, 1218, 1219, 1220, 2120, 2121, 2122, 2123, 2124, 1614, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2263, 2265, 2264, 2266, 2295, 763]
        # })
        # power_df = pd.merge(power_df, maks_units, how='outer', on='unit_id', indicator=True)
        # power_df = power_df.query('_merge == "both"')
        lignite_cost = 1.76
        # series_values = self.get_tables('bi', 'series_data')
        # powerunit_df = self.get_tables('im', 'power_powerunit_info')
        lignite_coef = {5: 0.4069, 6: 0.4069, 7: 0.4069, 10: 0.4128, 12: 0.3429,
                        15: 0.3439, 18: 0.4802, 31: 1, 33: 1, 32: 1, 35: 1, 34: 1,
                        27: 0.3469, 30: 0.4303, 41: 0.3734, 36: 0.6620}
        tickers = lignite_coef.keys()
        market_ticker = pd.DataFrame(data={
            'm_id': [21, 21, 21, 13, 24, 6, 23, 33, 34, 56, 35, 36, 25, 3, 11, 12],
            'series_id': [5, 6, 7, 10, 12, 15, 18, 31, 33, 32, 35, 34, 27, 30, 41, 36]

        })
        series_values = pd.DataFrame()
        for t in tickers:
            series_value = self.get_the_furthest_year_of_product(t)
            series_values = pd.concat([series_values, series_value], ignore_index=True)

        series_values = pd.merge(series_values, market_ticker, how='outer', on='series_id', indicator=True)
        series_values = series_values[series_values['_merge'] == 'both']
        series_values.drop(columns='_merge')
        result_df = pd.DataFrame()
        for t in tickers:
            # print(t)
            one_ticker_df = pd.DataFrame()
            one_ticker_df['datetime'] = series_values.query('series_id ==@t')['d_date']
            one_ticker_df['value'] = series_values.query('series_id ==@t')['value']
            one_ticker_df['coef'] = [lignite_coef[t]] * len(series_values.query('series_id ==@t')['value'])
            one_ticker_df['gfc_val2'] = series_values.query('series_id ==@t')['value'] * lignite_coef[t]

            one_ticker_df['series_id'] = [t] * len(one_ticker_df['value'])
            one_ticker_df['m_id'] = series_values.query('series_id ==@t')['m_id']
            # one_ticker_df['iso_code'] = series_values.query('series_id ==@t')['iso_code']
            result_df = pd.concat([result_df, one_ticker_df])
        total_powerunits = pd.DataFrame()
        for country in set(market_ticker['m_id'].values):  # цикл по странам
            power_country_df = power_df.query('id == @country')
            value_country_df = result_df.query('m_id == @country')
            if power_country_df.empty == False:
                for powerunit in set(power_country_df['unit_id'].values):
                    logger.info(f"lignite calculating for the {powerunit} powerunit")
                    one_power_df = pd.DataFrame()
                    one_power_df['datetime'] = value_country_df['datetime']
                    one_power_df['powerunit_id'] = [powerunit] * len(value_country_df['datetime'])
                    one_power_df['country'] = [country] * len(value_country_df['datetime'])
                    one_power_df['effectiveness'] = [power_country_df.query('unit_id == @powerunit')[
                                                         'effectiveness'].values[0]] * len(value_country_df['datetime'])
                    one_power_df['gfc_val8'] = [1 / one_power_df['effectiveness'].values[0] * 0.40709] * len(
                        value_country_df['datetime'])
                    one_power_df['series_id'] = value_country_df['series_id']
                    one_power_df['value'] = value_country_df['value']
                    one_power_df['gfc_val2'] = value_country_df['gfc_val2']
                    one_power_df['gfc_val3'] = one_power_df['gfc_val2'] * one_power_df['gfc_val8']
                    one_power_df['iso_code'] = [power_country_df['iso_code'].values[0]] * len(
                        value_country_df['datetime'])
                    one_power_df = one_power_df[(one_power_df['datetime'] >= self.first_forecast_date) & (one_power_df['datetime'] <= self.last_forecast_date)]
                    # one_power_df = one_power_df[one_power_df['datetime'] <= self.last_forecast_date]
                    self.get_df_per_hour(one_power_df, lignite_cost, scenario, powerunit, 1)
                    # total_powerunits = pd.concat([total_powerunits, one_power_df], ignore_index=True)
                    # total_powerunits = total_powerunits[
                    #     (total_powerunits['datetime'] >= self.first_forecast_date) &
                    #     (total_powerunits['datetime'] <= self.last_forecast_date)]
                    # total_powerunits = total_powerunits[total_powerunits['datetime'] >= first_forecast_date]
                    # total_powerunits = total_powerunits[total_powerunits['datetime'] <= last_forecast_date]

        logger.info('lignite was done successfully')
        # total_powerunits = reduce_memory_usage(total_powerunits)
        # return total_powerunits

    def coal(self, scenario):
        logger.info(f'running coal with {scenario} scenario')
        coal_cost = 1.76
        # first_forecast_date, last_forecast_date = self.get_first_last_forecast_date()
        power_df = self.get_powerunit_country(6)
        # maks_units = pd.DataFrame(data={
        #     'unit_id': [175, 176, 177, 1713, 1218, 1219, 1220, 2121, 2122, 2123, 1614, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2263, 2265, 2266, 2264, 2295, 763]
        # })
        # power_df = pd.merge(power_df, maks_units, how='outer', on='unit_id', indicator=True)
        # power_df = power_df.query('_merge == "both"')
        # power_df = self.get_powerunits_country().query('generation_type_id == 6 and unit_id == 2049')

        # powerunit_df = self.get_tables('im', 'power_powerunit_info')
        coal_coef_base = {3: 0.1639, 8: 0.1434, 11: 0.1434, 14: 0.1434,
                          17: 0.1565, 25: 0.1434, 28: 0.1434, 39: 0.1434,
                          44: 0.1434, 37: 0.1434, 42: 0.1434, 46: 0.1434,
                          23: 0.1434, 83: 0.1434, 82: 0.1434, 84: 0.1434
                          }
        coal_coef_low = {4: 0.1639, 9: 0.1434, 13: 0.1434, 16: 0.1434,
                         19: 0.1565, 26: 0.1434, 29: 0.1434, 40: 0.1434,
                         45: 0.1434, 38: 0.1434, 43: 0.1434, 47: 0.1434, 24: 0.1434}

        coal_coef_high = {48: 0.1639, 49: 0.1434, 50: 0.1434, 51: 0.1434,
                          52: 0.1565, 54: 0.1434, 55: 0.1434, 57: 0.1434,
                          59: 0.1434, 56: 0.1434, 58: 0.1434, 60: 0.1434, 53: 0.1434}
        market_ticker = pd.DataFrame(data={
            'm_id': [21, 21, 21, 13, 13, 13, 24, 24, 24, 6, 6, 6, 23, 23, 23, 25, 25, 25, 3, 3, 3, 11, 11, 11, 26, 26,
                     26, 15, 15, 15, 20, 20, 20, 22, 22, 22, 4, 4, 4, 7, 10, 55],
            'series_id': [3, 48, 4, 8, 49, 9, 11, 50, 13, 14, 51, 16, 17, 52, 19, 25, 54, 26, 28, 55, 29, 39,
                          57, 40, 44, 59, 45, 37, 56, 38, 42, 58, 53, 46, 60, 47, 23, 53, 24, 83, 82, 84]})
        if scenario == 1:
            coal_coef = coal_coef_base
        elif scenario == 2:
            coal_coef = coal_coef_high
        elif scenario == 3:
            coal_coef = coal_coef_low
        else:
            coal_coef = coal_coef_base

        tickers = coal_coef.keys()
        result_df = pd.DataFrame()
        series_values = pd.DataFrame()
        for t in tickers:
            series_value = self.get_the_furthest_year_of_product(t)
            series_values = pd.concat([series_values, series_value], ignore_index=True)
        series_values = pd.merge(series_values, market_ticker, how='outer', on='series_id', indicator=True)
        series_values = series_values[series_values['_merge'] == 'both']
        series_values.drop(columns='_merge')
        for t in tickers:
            # print(t)
            one_ticker_df = pd.DataFrame()
            one_ticker_df['datetime'] = series_values.query('series_id ==@t')['d_date']
            one_ticker_df['value'] = series_values.query('series_id ==@t')['value']
            one_ticker_df['coef'] = [coal_coef[t]] * len(series_values.query('series_id ==@t')['value'])
            one_ticker_df['gfc_val2'] = series_values.query('series_id ==@t')['value'] * coal_coef[t]
            one_ticker_df['series_id'] = [t] * len(one_ticker_df['value'])
            one_ticker_df['m_id'] = series_values.query('series_id ==@t')['m_id']
            result_df = pd.concat([result_df, one_ticker_df])
        total_powerunits = pd.DataFrame()
        for country in set(market_ticker['m_id'].values):  # цикл по странам
            power_country_df = power_df.query('id == @country')
            value_country_df = result_df.query('m_id == @country')
            # if country == 1 or country == 11 or country ==
            if power_country_df.empty == False:
                for powerunit in set(power_country_df['unit_id'].values):  # цикл по энергоблокам в стране
                    logger.info(f"coal calculating for the {powerunit} powerunit")
                    one_power_df = pd.DataFrame()
                    one_power_df['datetime'] = value_country_df['datetime']
                    one_power_df['powerunit_id'] = [powerunit] * len(value_country_df['datetime'])
                    one_power_df['country'] = [country] * len(value_country_df['datetime'])
                    one_power_df['series_id'] = value_country_df['series_id']
                    one_power_df['value'] = value_country_df['value']
                    one_power_df['effectiveness'] = [power_country_df.query('unit_id == @powerunit')[
                                                         'effectiveness'].values[0]] * len(value_country_df['datetime'])
                    one_power_df['gfc_val8'] = [1 / one_power_df['effectiveness'].values[0] * 0.14967] * len(
                        value_country_df['datetime'])
                    one_power_df['gfc_val2'] = value_country_df['gfc_val2']
                    one_power_df['gfc_val3'] = one_power_df['gfc_val2'] * one_power_df['gfc_val8']
                    one_power_df['iso_code'] = [power_country_df['iso_code'].values[0]] * len(
                        value_country_df['datetime'])
                    one_power_df = one_power_df[one_power_df['datetime'] >= self.first_forecast_date]
                    one_power_df = one_power_df[one_power_df['datetime'] <= self.last_forecast_date]
                    self.get_df_per_hour(one_power_df, coal_cost, scenario, powerunit, 1)
                    # print(one_power_df)
                    total_powerunits = pd.concat([total_powerunits, one_power_df], ignore_index=True)
                    total_powerunits = reduce_memory_usage(total_powerunits)
                    total_powerunits = total_powerunits[
                        (total_powerunits['datetime'] >= self.first_forecast_date) &
                        (total_powerunits['datetime'] <= self.last_forecast_date)]
                    # total_powerunits = total_powerunits[total_powerunits['datetime'] <= last_forecast_date]
                    # total_powerunits = total_powerunits[total_powerunits['datetime'] >= first_forecast_date]
        logger.info('coal was done successfully for powerunit')
        return total_powerunits

    def gas(self, scenario):
        logger.info(f'running gas with {scenario} scenario')
        gas_cost = 1.8436
        # first_forecast_date, last_forecast_date = self.get_first_last_forecast_date()
        power_df = self.get_powerunit_country(2)
        m_id = [21, 13, 6, 23, 25, 3, 4, 33, 34, 56, 35, 36, 12, 41, 20, 7, 37, 27, 31, 24, 11, 26, 22, 15, 10, 55, 2]
        if scenario == 1:
            market_ticker = pd.DataFrame(data={
                'm_id': m_id,
                'series_id': [21] * len(m_id)})

        elif scenario == 2:
            market_ticker = pd.DataFrame(data={
                'm_id': m_id,
                'series_id': [63] * len(m_id)})
            # print(market_ticker)
        else:
            market_ticker = pd.DataFrame(data={
                'm_id': m_id,
                'series_id': [62] * len(m_id)})

        tickers = set(market_ticker['series_id'].values)
        series_values = pd.DataFrame()
        for t in tickers:
            series_value = self.get_the_furthest_year_of_product(t)
            series_values = pd.concat([series_values, series_value], ignore_index=True)
        series_values = pd.merge(series_values, market_ticker, how='outer', on='series_id', indicator=True)
        series_values = series_values[series_values['_merge'] == 'both']
        series_values.drop(columns='_merge')
        # print(series_values)
        result_df = pd.DataFrame()
        for t in tickers:
            one_ticker_df = pd.DataFrame()
            one_ticker_df['datetime'] = series_values.query('series_id ==@t')['d_date']
            one_ticker_df['value'] = series_values.query('series_id ==@t')['value']
            one_ticker_df['gfc_val2'] = series_values.query('series_id ==@t')['value'] + 0.3
            one_ticker_df['series_id'] = [t] * len(one_ticker_df['value'])
            one_ticker_df['m_id'] = series_values.query('series_id ==@t')['m_id']
            result_df = pd.concat([result_df, one_ticker_df])
        total_powerunits = pd.DataFrame()
        for country in set(market_ticker['m_id'].values):  # цикл по странам
            power_country_df = power_df.query('id == @country')
            value_country_df = result_df.query('m_id == @country')
            if power_country_df.empty == False:
                for powerunit in set(power_country_df['unit_id'].values):  # цикл по энергоблокам в стране
                    logger.info(f"coal calculating for the {powerunit} powerunit")
                    one_power_df = pd.DataFrame()
                    one_power_df['datetime'] = value_country_df['datetime']
                    one_power_df['powerunit_id'] = [powerunit] * len(value_country_df['datetime'])
                    one_power_df['country'] = [country] * len(value_country_df['datetime'])
                    one_power_df['series_id'] = value_country_df['series_id']
                    one_power_df['value'] = value_country_df['value']
                    one_power_df['effectiveness'] = [power_country_df.query('unit_id == @powerunit')[
                                                         'effectiveness'].values[0]] * len(value_country_df['datetime'])
                    one_power_df['gfc_val8'] = [1 / one_power_df['effectiveness'].values[0]] * len(
                        value_country_df['datetime'])
                    one_power_df['gfc_val2'] = value_country_df['gfc_val2']
                    one_power_df['gfc_val3'] = one_power_df['gfc_val2'] * one_power_df['gfc_val8']
                    one_power_df['iso_code'] = [power_country_df['iso_code'].values[0]] * len(
                        value_country_df['datetime'])
                    one_power_df = one_power_df[one_power_df['datetime'] >= self.first_forecast_date]
                    one_power_df = one_power_df[one_power_df['datetime'] <= self.last_forecast_date]
                    self.get_df_per_hour(one_power_df, gas_cost, scenario, powerunit, 1)
                    # print(one_power_df)
                    total_powerunits = pd.concat([total_powerunits, one_power_df], ignore_index=True)
                    total_powerunits = total_powerunits[
                        (total_powerunits['datetime'] >= self.first_forecast_date) &
                        (total_powerunits['datetime'] <= self.last_forecast_date)]
                    # total_powerunits = total_powerunits[total_powerunits['datetime'] >= first_forecast_date & total_powerunits['datetime'] <= last_forecast_date]
                    # total_powerunits = total_powerunits[total_powerunits['datetime'] <= last_forecast_date]
                    total_powerunits = reduce_memory_usage(total_powerunits)
                    logger.info('gas was done successfully')
        return total_powerunits

    def gas_base_backtest(self, scenario):
        logger.info('running gas')
        gas_cost = 1.8436
        power_df = self.get_powerunit_country(2)
        gas_price_df = self.get_tables_iso('im', 'im_markets_forecast_calc_daily')
        market_country_df = self.get_tables_iso('im', 'im_market_country where m_commodity = 2')
        c_id = [21, 13, 6, 23, 25, 3, 4, 34, 35, 56, 35, 36, 12, 41, 20, 7, 37, 27, 31, 24, 11, 26, 22, 15, 10, 28, 2]
        m_id = [70, 66, 62, 71, 60, 61, 77, 58, 58, 58, 58, 58, 58, 58, 69, 63, 69, 69, 76, 72, 65, 64, 64, 67, 64, 74,
                59]
        result_df = pd.DataFrame()
        for market in m_id:
            # if market==67:
            daily_df = pd.DataFrame()
            hourly_df = pd.DataFrame()
            daily_df['datetime'] = gas_price_df.query('mfc_market_id == @market')['mfc_date']
            daily_df['gfc_val2'] = gas_price_df.query('mfc_market_id == @market')['mfc_val_1']
            daily_df['c_id'] = [market_country_df.query("m_id == @market")['m_country'].values[0]] * len(
                gas_price_df.query('mfc_market_id == @market')['mfc_date'])
            daily_df['m_id'] = [market] * len(gas_price_df.query('mfc_market_id == @market')['mfc_date'])
            result_df = pd.concat([result_df, daily_df], ignore_index=True)
            # print(m_id)
        total_powerunits = pd.DataFrame()
        for country in c_id:  # цикл по странам
            power_country_df = power_df.query('id == @country')
            value_country_df = result_df.query('c_id == @country')
            if power_country_df.empty == False and value_country_df.empty == False:
                for powerunit in set(power_country_df['unit_id'].values):  # цикл по энергоблокам в стране
                    logger.info(f"gas calculating for the {powerunit} powerunit")
                    one_power_df = pd.DataFrame()
                    one_power_df['datetime'] = value_country_df['datetime']
                    one_power_df['powerunit_id'] = [powerunit] * len(value_country_df['datetime'])
                    one_power_df['country'] = [country] * len(value_country_df['datetime'])
                    # one_power_df['series_id'] = value_country_df['series_id']
                    # one_power_df['value'] = value_country_df['value']
                    one_power_df['effectiveness'] = [power_country_df.query('unit_id == @powerunit')[
                                                         'effectiveness'].values[0]] * len(value_country_df['datetime'])
                    one_power_df['gfc_val8'] = [1 / one_power_df['effectiveness'].values[0]] * len(
                        value_country_df['datetime'])
                    one_power_df['gfc_val2'] = value_country_df['gfc_val2']
                    one_power_df['gfc_val3'] = one_power_df['gfc_val2'] * one_power_df['gfc_val8']
                    one_power_df['iso_code'] = [power_country_df['iso_code'].values[0]] * len(
                        value_country_df['datetime'])
                    one_power_df = one_power_df[one_power_df['datetime'] >= self.first_forecast_date.date()]
                    one_power_df = one_power_df[one_power_df['datetime'] <= self.last_forecast_date.date()]
                    self.get_df_per_hour(one_power_df, gas_cost, scenario, powerunit, 1)
                    # print(one_power_df)
                    total_powerunits = pd.concat([total_powerunits, one_power_df], ignore_index=True)
                    total_powerunits = total_powerunits[
                        (total_powerunits['datetime'] >= self.first_forecast_date.date()) &
                        (total_powerunits['datetime'] <= self.last_forecast_date.date())]
                    # total_powerunits = total_powerunits[total_powerunits['datetime'] >= first_forecast_date.date() & total_powerunits['datetime'] <= last_forecast_date.date()]
                    # total_powerunits = total_powerunits[total_powerunits['datetime'] <= last_forecast_date.date()]
                    total_powerunits = reduce_memory_usage(total_powerunits)
                    logger.info('gas was done successfully')
        total_powerunits = total_powerunits.drop_duplicates(subset=['datetime'], keep='last')
        return total_powerunits

    def co2_for_past_periods(self):  # 242-31

        m_id = [21, 13, 24, 6, 23, 4, 10, 25, 3, 11, 12, 37, 20, 2, 15, 26, 22, 7, 27]
        market_ticker = pd.DataFrame(data={
            'm_id': m_id,
            'series_id': [20] * len(m_id)})
        market_ticker_new = pd.DataFrame(data={'m_id': [31], 'series_id': [80]})
        market_ticker = pd.concat([market_ticker, market_ticker_new], ignore_index=True)
        # print(market_ticker)
        tickers = set(market_ticker['series_id'].values)
        series_values = pd.DataFrame()
        past_co2_df = pd.DataFrame()
        for t in tickers:
            series_value = self.get_the_furthest_year_of_product(t)
            series_values = pd.concat([series_values, series_value], ignore_index=True)
            date = datetime.now()
            series_values = series_values[series_values['d_date'] <= date]
            past_co2 = self.co2(series_values, market_ticker)
            past_co2_df = pd.concat([past_co2_df, past_co2], ignore_index=True)
        past_co2_df = past_co2_df[past_co2_df['_merge'] == 'both']
        past_co2_df.fillna(0, inplace=True)
        # past_co2_df = past_co2_df[past_co2_df['d_date']>='2021-01-01']
        return past_co2_df

    def co2_for_future_periods(self, scenario):  # 242-31
        m_id = [21, 13, 24, 6, 23, 4, 10, 25, 3, 11, 12, 37, 20, 2, 15, 26, 22, 7, 27]
        if scenario == 2:
            s_id = 77
        elif scenario == 3:
            s_id = 78
        else:
            s_id = 22
        market_ticker = pd.DataFrame(data={
            'm_id': m_id,
            'series_id': [s_id] * len(m_id)})
        market_ticker_new = pd.DataFrame(data={'m_id': [31], 'series_id': [80]})
        market_ticker = pd.concat([market_ticker, market_ticker_new], ignore_index=True)
        # print(market_ticker)
        tickers = set(market_ticker['series_id'].values)
        series_values = pd.DataFrame()
        for t in tickers:
            series_value = self.get_the_furthest_year_of_product(t)
            series_values = pd.concat([series_values, series_value], ignore_index=True)
        date = datetime.now()
        series_values = series_values[series_values['d_date'] >= date]
        future_co2_df = self.co2(series_values, market_ticker)
        future_co2_df = future_co2_df[future_co2_df['_merge'] == 'both']
        future_co2_df.fillna(0, inplace=True)
        return future_co2_df



    def co2(self, series_values, market_ticker):
        series_values = pd.merge(series_values, market_ticker, how='outer', on='series_id', indicator=True)
        series_values = series_values[series_values['_merge'] == 'both']
        series_values.drop(columns='_merge')
        result_df = pd.DataFrame()
        countries = list(set(market_ticker['m_id'].values))
        i = 0
        lig_df = self.get_powerunit_country(7)
        coal_df = self.get_powerunit_country(6)
        gas_df = self.get_powerunit_country(2)
        while i < len(countries):
            # for t in countries:
            t = countries[i]

            # print(t)
            one_ticker_df = pd.DataFrame()
            one_ticker_df['series_id'] = series_values.query('m_id ==@t')['series_id']
            if one_ticker_df.empty == False:
                power_df = pd.concat(
                    [gas_df, coal_df , lig_df],
                    join='outer', ignore_index=True)
                power_df = power_df.rename(columns={'id': 'm_id'})
                serie_id = one_ticker_df['series_id'].values[0]
                one_ticker_df['datetime'] = series_values.query('series_id ==@serie_id')['d_date']
                one_ticker_df['m_id'] = [t] * len(one_ticker_df['datetime'])
                power_df = power_df[power_df['m_id'] == t]
                if power_df.empty == False:
                    power_df['for_model'].fillna(0, inplace=True)
                    one_ticker_df = pd.merge(one_ticker_df, power_df[['unit_id', 'm_id', 'for_model']], how='outer',
                                             on='m_id',
                                             indicator=True)
                    one_ticker_df = one_ticker_df[one_ticker_df['_merge'] == 'both']
                else:
                    one_ticker_df['for_model'] = [0] * len(one_ticker_df['datetime'])
                one_ticker_df['value'] = series_values.query('m_id == @serie_id')['value']

                one_ticker_df['gfc_val5'] = series_values.query('series_id ==@serie_id')['value'] \
                                            * 0.98 * one_ticker_df['for_model']
                result_df = pd.concat([result_df, one_ticker_df], ignore_index=True)
                result_df = result_df[result_df.value.notnull()]

                i += 1
            else:

                i += 1

        return result_df

    def get_last_iteration(self):
        iteration_df = self.get_tables('im', 'im_iteration')
        iteration = max(iteration_df['iteration_id'].values)
        # logger.info(f'got last iteration {iteration}')
        return iteration



    def get_df_per_hour(self, commodity_df, commodity_cost, scenario, powerunit, commodity_id=1):
        logger.info('starting to prepare main data to insert into table')
        powerunits = list(set(commodity_df['powerunit_id'].values))
        co2_df_past = self.co2_for_past_periods()
        co2_df_future = self.co2_for_future_periods(scenario)
        co2_df = pd.concat([co2_df_past, co2_df_future], ignore_index=True)
        total_df = pd.DataFrame()
        final_df = pd.DataFrame()
        powerunit = int(powerunit)
        # i=0
        # while i<=len(powerunits):
        # # for powerunit in powerunits:
        #     powerunit = powerunits[i]
        df = pd.DataFrame()
        logger.info(f"collecting data from {powerunit} powerunit")
        # co2_df = co2_df.query('unit_id == @powerunit')
        commodity_df = commodity_df.query('powerunit_id == @powerunit').drop_duplicates(subset=['datetime'],
                                                                                        keep='last')
        for i in range(len(commodity_df) - 1):
            try:
                if commodity_df['gfc_val2'].values[i + 1] == 'nan':
                    commodity_df['gfc_val2'].values[i + 1] = commodity_df['gfc_val2'].values[i]
                if commodity_df['gfc_val3'].values[i + 1] == 'nan':
                    commodity_df['gfc_val3'].values[i + 1] = commodity_df['gfc_val3'].values[i]
            except Exception:
                pass
        dates = sorted(pd.to_datetime(commodity_df['datetime']))
        co2_df = pd.concat([co2_df_past, co2_df_future], ignore_index=True)
        for d in range(0, len(dates) - 1):
            main_df = pd.DataFrame()
            da = [dates[d]]

            delta = dates[d + 1] - dates[d]
            delta = int(f"{delta}".split(' ')[0]) * 24
            # print(delta)
            # da = [da[t - 1] + timedelta(hours=1) for t in range(1, delta)]
            for t in range(1, delta):
                da.append(da[t - 1] + timedelta(hours=1))
            try:
                x = dates[d]
                gfc_val2 = commodity_df.query("powerunit_id == @powerunit") \
                    [commodity_df['datetime'] == x]['gfc_val2'].values[0]
                gfc_val8 = commodity_df.query("powerunit_id == @powerunit") \
                    [commodity_df['datetime'] == x]['gfc_val8'].values[0]
                gfc_val3 = commodity_df.query("powerunit_id == @powerunit") \
                    [commodity_df['datetime'] == x]['gfc_val3'].values[0]
                country = commodity_df.query("powerunit_id == @powerunit") \
                    [commodity_df['datetime'] == x]['country'].values[0]
                cur = self.__connection()
                cur.execute(f"SELECT m_id FROM im.im_market_country "
                            f"WHERE m_commodity = {commodity_id} AND m_country = {country}")
                mfc_market_id = cur.fetchall()[0][0]
                # print(co2_df)
                try:
                    gfc_val4_df = co2_df.query('unit_id == @powerunit') \
                        [co2_df['datetime'] == x]['value']
                    gfc_val5_df = co2_df.query('unit_id == @powerunit') \
                        [co2_df['datetime'] == x]['gfc_val5']
                    if gfc_val4_df.empty == False:
                        gfc_val4 = gfc_val4_df.values[0]
                    else:
                        gfc_val4_df = co2_df.query('unit_id == @powerunit') \
                            [co2_df['datetime'] <= x]['value']
                        gfc_val4 = gfc_val4_df.values[-1]

                    if gfc_val5_df.empty == False:
                        gfc_val5 = gfc_val5_df.values[0]
                    else:
                        gfc_val5_df = co2_df.query('unit_id == @powerunit') \
                            [co2_df['datetime'] <= x]['gfc_val5']
                        gfc_val5 = gfc_val5_df.values[-1]
                except Exception as e:
                    gfc_val5 = 0
                    gfc_val4 = 0
                    logger.info(f"there is problem {e} with co2 with unit {powerunit}")

                gfc_val6 = commodity_cost
                try:
                    if co2_df.query("unit_id == @powerunit ")[co2_df['datetime'] == x]['for_model'].empty == False:
                        gfc_val7 = co2_df.query("unit_id == @powerunit ")[co2_df['datetime'] == x]['for_model'].values[
                            0]
                    else:
                        gfc_val7 = co2_df.query("unit_id == @powerunit ")[co2_df['datetime'] <= x]['for_model'].values[
                            -1]
                except Exception:
                    gfc_val7 = 0

                gfc_val8 = gfc_val8
                gfc_val1 = gfc_val3 + gfc_val5 + gfc_val6

                main_df['gfc_iteration'] = [self.iteration] * len(da)
                main_df['gfc_scenario'] = [scenario] * len(da)
                main_df['gfc_local_datetime'] = da
                main_df['gfc_utc_datetime'] = substract_time_shift(da,
                                                                   entsoe_configs.COUTRIES_SHIFTS[
                                                                       f"{commodity_df['iso_code'].values[0]}"][1],
                                                                   entsoe_configs.COUTRIES_SHIFTS[
                                                                       f"{commodity_df['iso_code'].values[0]}"][0])

                main_df['gfc_market_id'] = [mfc_market_id] * len(da)
                main_df['gfc_generationunit_id'] = [powerunit] * len(da)
                main_df['gfc_microservice_id'] = [1] * len(da)
                main_df['gfc_indicator_id'] = [14] * len(da)
                main_df['gfc_val_1'] = [gfc_val1] * len(da)
                main_df['gfc_val_2'] = [gfc_val2] * len(da)
                main_df['gfc_val_3'] = [gfc_val3] * len(da)
                main_df['gfc_val_4'] = [gfc_val4] * len(da)
                main_df['gfc_val_5'] = [gfc_val5] * len(da)
                main_df['gfc_val_6'] = [gfc_val6] * len(da)
                main_df['gfc_val_7'] = [gfc_val7] * len(da)
                main_df['gfc_val_8'] = [gfc_val8] * len(da)
            except Exception as e:
                logger.info(f"there is an error {e} with powerunit {powerunit}")
                break
            main_df.drop_duplicates(
                subset=['gfc_iteration', 'gfc_utc_datetime', 'gfc_scenario', 'gfc_generationunit_id',
                        'gfc_microservice_id'])
            df = pd.concat([df, main_df], ignore_index=True)
            main_df = reduce_memory_usage(main_df)
            if main_df.empty == True:
                logger.info(f"main dataframe is empty for the generationunit {powerunit}")
            # logger.info('duplicates were dropped')
            try:
                insert_into_table(main_df, 'im', 'im_generationunit_forecast_calc',
                                  custom_conflict_resolution='on conflict do nothing')
            except Exception as e:
                logger.info(f"there is a mistake while inserting unit {powerunit} like {e}")
        # total_df = pd.concat([total_df, df], ignore_index=True)
        logger.info(f'data was inserted into im_generationunit_forecast_calc table for generationunit = {powerunit} for scenario {scenario}')

    def run_commodities(self, scenario):

        lignite_df = self.lignite(scenario)
        coal_df = self.coal(scenario)
        if scenario == 1 or scenario == 4:
            gas_df = self.gas_base_backtest(scenario)
        elif scenario == 2 or scenario == 3:
            gas_df = self.gas(scenario)
        else:
            pass

    @start_end_microservice_time(1)
    def run_all_scenario(self):
        self.first_forecast_date, self.last_forecast_date = self.get_first_last_forecast_date()
        self.iteration = self.get_last_iteration()
        logger.info(f'got last iteration {self.iteration}')
        p = Pool()
        p1 = Pool()

        scenario_list = [1, 2, 3, 4]
        # p.map(self.lignite, scenario_list)
        # p.close()
        # p.join()
        # p1.map(self.coal, scenario_list)
        # p1.close()
        # p1.join()


        for scenario in scenario_list:
            p1 = Process(target=self.lignite, args=(scenario,))
            p2 = Process(target=self.coal, args=(scenario,))

            if scenario == 1 or scenario == 4:
                p3 = Process(target=self.gas_base_backtest, args=(scenario,))
                # p3.start()
                # p3.join()
            else:
                p3 = Process(target=self.gas, args=(scenario,))
                # p3.start()
                # p3.join()
            p1.start()
            p2.start()
            p3.start()
            p1.join()
            p2.join()
            p3.start()
            p3.join()

            logger.info(f'run scenario {scenario}')
            # self.run_commodities(scenario)
            logger.info(f"scenario {scenario} ended")
        logger.info('program ended')
