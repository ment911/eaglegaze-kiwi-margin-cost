from eaglegaze_common.common_utils import insert_into_table, start_end_microservice_time, \
    reduce_memory_usage, set_iteration_values, chunker, get_time_shift, get_countries
from eaglegaze_common.entsoe_configs import COUTRIES_SHIFTS
from eaglegaze_common.common_attr import Attributes as at
from eaglegaze_common import entsoe_configs, logger
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine
from decouple import config as envs

import concurrent.futures
import pandas as pd
import numpy as np
import warnings

import time
import sys

import os

warnings.filterwarnings("ignore")
logger = logger.get_logger(__name__, at.LogAttributes.log_file)

engine = create_engine(envs('ALCHEMY_CONNECTION', cast=str))
con = engine.raw_connection()
cur = con.cursor()


class MarginCost:
    def __init__(self):
        self.iteration, self.start_timestamp, self.end_timestamp = set_iteration_values()
        # self.iteration = 100
        self.microservice = 1
        self.indicator = 14
        self.var_cost = pd.DataFrame({'var_cost': ['coal', 'lignite', 'gas'],
                                      'price': [1.76, 1.76, 1.8436],
                                      'generation_type_id': [6, 7, 2],
                                      'heating_value': [149.67, 407.09, 75.]})
        self.all_series_data = None
        self.lignite_price_method = None
        self.coal_price_method = None
        self.gas_price_method = None
        self.co2_price_method_past = None
        self.co2_price_method_future = None

        self.series_data_lignite = None
        self.series_data_coal = None
        self.series_data_gas = None
        self.series_data_co2 = None
        self.series_scenarios()

        self.units = self.get_units()
        self.co2_rates = self.get_co2_rates()
        self.get_all_series()

    def get_all_series(self):
        cur.execute(f"select * from bi.series_data where d_date >= '{self.start_timestamp}' "
                    f"and d_date <= '{self.end_timestamp}';")
        self.all_series_data = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])

    def get_units(self):
        cur.execute(f'select unit_id, power_powerunit_info.eic_code as eic_code, generation_id, '
                    f'start_date, end_date, country, m_id from im.power_powerunit_info left join '
                    f'im.power_powerstation_info on power_powerunit_info.station_id = power_powerstation_info.id '
                    f'left join im.im_market_country on country = m_country where model_id = 1;')
        units = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        units = units[units['generation_id'].isin(self.var_cost['generation_type_id'].tolist())]
        return units

    def series_scenarios(self):
        lignite_price_method_1 = pd.DataFrame(
            {'m_id': [42, 42, 42, 38, 45, 33, 44, 51, 52, 56, 53, 54, 46, 31, 36, 37],
             'series_id': [5, 6, 7, 10, 12, 15, 18, 31, 32, 35, 33, 34, 27, 30, 41, 36],
             'coef': [0.4069, 0.4069, 0.4069, 0.4128, 0.3429, 0.3439, 0.4802, 1, 1, 1, 1, 1, 0.3469, 0.4303, 0.3734,
                      0.6620]})
        self.lignite_price_method = {1: lignite_price_method_1, 2: lignite_price_method_1, 3: lignite_price_method_1}

        coal_price_method_1 = pd.DataFrame({'m_id': [42, 38, 45, 33, 44, 46, 31, 36, 47, 39, 41, 43, 32, 34, 35, 49],
                                            'series_id': [3, 8, 11, 14, 17, 25, 28, 39, 44, 37, 42, 46, 23, 83, 82,
                                                          84],
                                            'coef': [0.1639, 0.1434, 0.1434, 0.1434, 0.1565, 0.1434, 0.1434, 0.1434,
                                                     0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434]})

        coal_price_method_2 = pd.DataFrame({'m_id': [42, 38, 45, 33, 44, 46, 31, 36, 47, 39, 41, 43, 32, 34, 35, 49],
                                            'series_id': [48, 49, 50, 51, 52, 54, 55, 57, 59, 56, 58, 60, 53, None,
                                                          None, None],
                                            'coef': [0.1639, 0.1434, 0.1434, 0.1434, 0.1565, 0.1434, 0.1434, 0.1434,
                                                     0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434]})
        coal_price_method_3 = pd.DataFrame({'m_id': [42, 38, 45, 33, 44, 46, 31, 36, 47, 39, 41, 43, 32, 34, 35, 49],
                                            'series_id': [4, 9, 13, 16, 19, 26, 29, 40, 45, 38, 53, 47, 24, None,
                                                          None, None],
                                            'coef': [0.1639, 0.1434, 0.1434, 0.1434, 0.1565, 0.1434, 0.1434, 0.1434,
                                                     0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434, 0.1434]})
        self.coal_price_method = {1: coal_price_method_1, 2: coal_price_method_2, 3: coal_price_method_3}

        gas_price_method_1 = pd.DataFrame({'m_id': [42, 38, 33, 44, 46, 31, 32, 51, 52, 56, 53, 54, 37, 57, 41, 34, 55,
                                                    48, 50, 45, 36, 47, 43, 39, 35, 49, 30],
                                           'series_id': [70, 66, 62, 71, 73, 60, 61, 77, 58, 58, 58, 58, 58, 58, 69,
                                                         63, 2, 69, 76, 72, 65, 64, 64, 67, 64, 74, 59],
                                           'coef': [1] * 27})
        gas_price_method_2 = gas_price_method_1.copy()
        gas_price_method_3 = gas_price_method_1.copy()
        gas_price_method_2['series_id'] = 63
        gas_price_method_3['series_id'] = 62
        self.gas_price_method = {1: gas_price_method_1, 2: gas_price_method_2, 3: gas_price_method_3}

        co2_price_method_1 = pd.DataFrame(
            {'m_id': [42, 38, 45, 33, 44, 31, 35, 46, 31, 36, 37, 55, 41, 30, 39, 47, 43, 34, 48, 49, 50],
             'series_id': ([20] * 19) + [85, 80],
             'coef': [1] * 21})
        co2_price_method_2 = co2_price_method_1.copy()
        co2_price_method_2['series_id'] = [78] * 21
        co2_price_method_3 = co2_price_method_1.copy()
        co2_price_method_3['series_id'] = [77] * 21
        self.co2_price_method_past = {1: co2_price_method_1, 2: co2_price_method_2, 3: co2_price_method_3}
        co2_price_method_1 = pd.DataFrame(
            {'m_id': [42, 38, 45, 33, 44, 31, 35, 46, 31, 36, 37, 55, 41, 30, 39, 47, 43, 34, 48, 49, 50],
             'series_id': ([76] * 19) + [85, 80],
             'coef': [1] * 21})
        co2_price_method_2 = co2_price_method_1.copy()
        co2_price_method_2['series_id'] = [78] * 21
        co2_price_method_3 = co2_price_method_1.copy()
        co2_price_method_3['series_id'] = [77] * 21
        self.co2_price_method_future = {1: co2_price_method_1, 2: co2_price_method_2, 3: co2_price_method_3}

    def fill_dfs(self, scenario):
        # LIGNITE
        self.series_data_lignite = self.all_series_data[
            self.all_series_data['series_id'].isin(self.lignite_price_method[scenario]['series_id'].unique())]
        # COAL
        self.series_data_coal = self.all_series_data[
            self.all_series_data['series_id'].isin(self.coal_price_method[scenario]['series_id'].unique())]
        # GAS
        self.series_data_gas = self.all_series_data[
            self.all_series_data['series_id'].isin(self.gas_price_method[scenario]['series_id'].unique())]
        # CO2
        series_data_co2_past = self.all_series_data[
            self.all_series_data['series_id'].isin(self.co2_price_method_past[scenario]['series_id'].unique())]
        series_data_co2_future = self.all_series_data[
            self.all_series_data['series_id'].isin(self.co2_price_method_past[scenario]['series_id'].unique())]
        self.series_data_co2 = series_data_co2_past.append(series_data_co2_future, ignore_index=True)
        self.series_data_co2 = self.series_data_co2.sort_values(by=['d_date'])
        self.series_data_co2 = self.series_data_co2.drop_duplicates()

    def spread_timestamps(self, unit_df):
        df = pd.DataFrame()
        df['gfc_utc_datetime'] = pd.date_range(start=self.start_timestamp, end=self.end_timestamp, freq='H')
        df = df[(df['gfc_utc_datetime'] >= pd.to_datetime(unit_df['start_date']).tolist()[0]) &
                (df['gfc_utc_datetime'] <= pd.to_datetime(unit_df['end_date']).tolist()[0])]
        return df

    def spread_values_for_dates(self, values_df, hourly_df, rename_value=None):
        if hourly_df is None or values_df is None:
            return pd.DataFrame()
        df = pd.merge(hourly_df, values_df, how='left', left_on=['gfc_utc_datetime', 'series_id'],
                      right_on=['d_date', 'series_id'])
        df = df.fillna(method='ffill').drop(columns=['frequency_id', 'd_date', 'sdt_id'])
        if rename_value:
            df.rename(columns={'value': rename_value}, inplace=True)
        return df

    def merge_prices(self, init_df, price_df, method_df, scenario):
        df = init_df.copy()
        df = pd.merge(df, method_df[scenario], left_on='gfc_market_id', right_on='m_id').drop(
            columns=['m_id'])
        df = self.spread_values_for_dates(values_df=price_df, hourly_df=df,
                                          rename_value='generation_series_price')
        return df

    def merge_co2(self, init_df, eic, scenario):
        if init_df.empty:
            return pd.DataFrame()
        df = init_df.copy()
        if df['gfc_utc_datetime'].min() <= datetime.today() <= df['gfc_utc_datetime'].max():
            df_1 = df[df['gfc_utc_datetime'] <= datetime.today()]
            df_1 = pd.merge(df_1, self.co2_price_method_past[scenario], how='left', left_on='gfc_market_id',
                            right_on='m_id')
            df_1 = df_1.rename(columns={'series_id_x': 'series_id', 'coef_x': 'coef',
                                        'series_id_y': 'series_id_co2', 'coef_y': 'coef_co2'})

            df_2 = df[df['gfc_utc_datetime'] > datetime.today()]
            df_2 = pd.merge(df_2, self.co2_price_method_future[scenario], how='left', left_on='gfc_market_id',
                            right_on='m_id')
            df_2 = df_2.rename(columns={'series_id_x': 'series_id', 'coef_x': 'coef',
                                        'series_id_y': 'series_id_co2', 'coef_y': 'coef_co2'})

            df = df_1.append(df_2, ignore_index=True)
            df.drop(columns=['m_id'], inplace=True)
        df['series_id'] = df['series_id'].astype(int)
        if 'series_id_co2' in df.columns:
            try:
                df = pd.merge(df, self.series_data_co2, how='left', left_on=['series_id_co2', 'gfc_utc_datetime'],
                              right_on=['series_id', 'd_date'])
                df = df.drop(columns=['series_id_y', 'frequency_id', 'd_date', 'sdt_id'])
                df = df.rename(columns={'series_id_x': 'series_id'})
                df['value'] = df['value'].fillna(method='ffill')
                df = df.dropna()
                df = df.rename(columns={'value': 'gfc_val_4'})
            except IndexError:
                print(df)
                sys.exit(1)
            except ValueError:
                pass
        if eic in self.co2_rates['powerunit_eic_code'].values:
            if 'gfc_val_4' in df.columns:
                df['gfc_val_5'] = df['gfc_val_4'] * \
                                  self.co2_rates[self.co2_rates['powerunit_eic_code'] == eic]['for_model'].values[
                                      0] * 0.98
            else:
                df['gfc_val_5'] = 0
            df['gfc_val_7'] = self.co2_rates[self.co2_rates['powerunit_eic_code'] == eic]['for_model'].values[0]
        return df

    def rename_and_rebase(self, init_df):
        if init_df.empty:
            return pd.DataFrame()
        df = init_df.copy()
        df['generation_series_price'] = df['generation_series_price'] * df['coef'] + 0.3

        gen_id = self.units[self.units['unit_id'] == df['gfc_generationunit_id'].values[0]]['generation_id'].values[0]
        df['gfc_val_8'] = self.var_cost[self.var_cost['generation_type_id'] == gen_id]['heating_value'].values[0]
        df['gfc_val_6'] = self.var_cost[self.var_cost['generation_type_id'] == gen_id]['price'].values[0]
        df['gfc_val_3'] = df['gfc_val_8'] * df['generation_series_price']
        df = df.rename(columns={'generation_series_price': 'gfc_val_2'})
        df['gfc_val_1'] = df['gfc_val_3'] + df['gfc_val_6']
        if 'gfc_val_5' in df.columns:
            df['gfc_val_1'] += df['gfc_val_5']
        df = pd.merge(df, get_countries(1), how='left', left_on='gfc_market_id', right_on='market_id')
        df['gfc_microservice_id'] = self.microservice
        df['gfc_indicator_id'] = self.indicator
        df['gfc_local_datetime'] = get_time_shift(df['gfc_utc_datetime'].tolist(),
                                                  COUTRIES_SHIFTS[df['iso_code'].tolist()[0]][1],
                                                  COUTRIES_SHIFTS[df['iso_code'].tolist()[0]][0])
        if 'gfc_val_5' in df.columns and 'gfc_val_4' in df.columns:
            df = df[['gfc_scenario', 'gfc_utc_datetime', 'gfc_local_datetime', 'gfc_market_id', 'gfc_generationunit_id',
                     'gfc_microservice_id', 'gfc_indicator_id', 'gfc_val_1', 'gfc_val_2', 'gfc_val_3', 'gfc_val_4',
                     'gfc_val_5', 'gfc_val_6', 'gfc_val_7', 'gfc_val_8']]
        elif 'gfc_val_4' in df.columns:
            df = df[['gfc_scenario', 'gfc_utc_datetime', 'gfc_local_datetime', 'gfc_market_id', 'gfc_generationunit_id',
                     'gfc_microservice_id', 'gfc_indicator_id', 'gfc_val_1', 'gfc_val_2', 'gfc_val_3', 'gfc_val_4',
                     'gfc_val_6', 'gfc_val_8']]
        else:
            df = df[['gfc_scenario', 'gfc_utc_datetime', 'gfc_local_datetime', 'gfc_market_id', 'gfc_generationunit_id',
                     'gfc_microservice_id', 'gfc_indicator_id', 'gfc_val_1', 'gfc_val_2', 'gfc_val_3',
                     'gfc_val_6', 'gfc_val_8']]
        df['gfc_iteration'] = self.iteration
        df = reduce_memory_usage(df, False)
        return df

    def commodity_iter(self, init_df, gen_id, scenario):
        if gen_id == 2:
            df_prices = self.series_data_gas
            df_method = self.gas_price_method
        elif gen_id == 6:
            df_prices = self.series_data_coal
            df_method = self.coal_price_method
        elif gen_id == 7:
            df_prices = self.series_data_coal
            df_method = self.lignite_price_method
        df = self.merge_prices(init_df, df_prices, df_method, scenario)
        return df

    def iterate_unit(self, unit_id):
        unit = self.units[self.units['unit_id'] == unit_id]
        df_out = pd.DataFrame()
        for scenario in (1, 2, 3):
            df = self.spread_timestamps(unit)
            df['gfc_generationunit_id'] = unit_id
            df['gfc_market_id'] = unit['m_id'].tolist()[0]
            self.fill_dfs(scenario)
            df['gfc_scenario'] = scenario
            df = self.commodity_iter(df, unit['generation_id'].tolist()[0], scenario)
            df = self.merge_co2(df, unit['eic_code'].tolist()[0], scenario)
            df = self.rename_and_rebase(df)
            df_out = df_out.append(df, ignore_index=True)
        # logger.info(unit_id)
        df_out.drop_duplicates(subset=['gfc_iteration', 'gfc_scenario', 'gfc_utc_datetime',
                                       'gfc_generationunit_id', 'gfc_microservice_id'], inplace=True)
        return df_out

    @start_end_microservice_time(1)
    def run(self):
        start_time = time.time()
        logger.info(f'{datetime.now()} - Started margin cost microservice. \n')
        df = pd.DataFrame()
        N = len(self.units['unit_id'])
        iter_var = 0
        step = 100
        while iter_var < N:
            start_iter = time.time()
            logger.info(f'Started from {iter_var}')
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(self.iterate_unit, self.units['unit_id'][iter_var:iter_var + step])
                for result in results:
                    if result is not None and not result.empty:
                        df = df.append(result, ignore_index=True)
            logger.info(f'{datetime.now()} - Inserting len = {len(df)}...')
            for chunk in chunker(df, 100000):
                insert_into_table(chunk, shema_name='im2',
                                  table_name='im_generationunit_forecast_calc',
                                  constraint='im_generationunit_forecast_ca_gfc_iteration_gfc_scenario_gf_key',
                                  primary_key=False)
            df = df.iloc[0:0]
            iter_var += step
            logger.info(f'Finished in {round((time.time() - start_iter), 1)} seconds.')
            # self.series_data_lignite = None
            # self.series_data_coal = None
            # self.series_data_gas = None
            # self.series_data_co2 = None
        '''
        for unit_id in self.units['unit_id']:
            a = time.time()
            res = self.iterate_unit(unit_id)
            df = df.append(res, ignore_index=True)
            if len(df) > 1_000_000:
                logger.info(f'Inserting on {iter_var}')
                #for chunk in chunker(df, 100000):
                insert_into_table(df, shema_name='im2',
                                  table_name='im_generationunit_forecast_calc',
                                  constraint='im_generationunit_forecast_ca_gfc_iteration_gfc_scenario_gf_key',
                                  primary_key=False)
                df = df.iloc[0:0]
            iter_var += 1
            # if iter_var % 100 == 0:
            logger.info(f'Done {iter_var} / {N} in {round(time.time() - a, 2)}')
        '''
        logger.info(f'Finished in {round((time.time() - start_time), 1)} seconds.')

    @staticmethod
    def get_co2_rates():
        cur.execute('select powerunit_eic_code, for_model from im.power_co2_info where for_model is not null;')
        rates = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        return rates


if __name__ == '__main__':
    MarginCost().run()
