import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from decouple import config as envs
from sqlalchemy import create_engine
from eaglegaze_common import  entsoe_configs, logger
from eaglegaze_common.common_attr import Attributes as at
from eaglegaze_common.common_utils import insert_into_table, substract_time_shift
logger = logger.get_logger(__name__, at.LogAttributes.log_file)


class Margin_cost():
    def connection(self):
        engine = create_engine(envs('ALCHEMY_CONNECTION', cast=str))
        con = engine.raw_connection()
        cur = con.cursor()
        return cur

    def get_tables(self, schema, table_name):
        cur = self.connection()
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
    
    def get_powerunits_country(self):
        powerunit_data = self.get_tables_iso('im', 'power_powerunit_info')
        powerunit_info = self.get_tables('bi', 'power_unit_info_entsoe')
        powerunit_co2 = self.get_tables_iso('im', 'power_co2_info')
        countries = self.get_tables('bi', 'countries')
        powerunit_co2 = powerunit_co2.rename(columns={'powerunit_eic_code': 'eic_code'})
        powerunit_info_co2 = pd.merge(powerunit_info, powerunit_co2, how='outer', on='eic_code', indicator=True)
        # powerunit_info_co2 = powerunit_info[powerunit_info['_merge'] == 'both']
        powerunit_info_co2 = powerunit_info_co2.query("_merge == 'both'")[['unit_name', 'eic_code', 'country_code', 'generation_type_id', 'tso', 'for_model']]
        power_df = pd.merge(powerunit_data, powerunit_info_co2, how='outer', on='eic_code', indicator=True)
        power_df = power_df[power_df['_merge'] == 'both']
        # power_df = pd.merge(power_df, powerunit_co2, how='outer', on='eic_code', indicator=True)
        # power_df = power_df[power_df['_merge'] == 'both']
        power_df = power_df.rename(columns = {'country_code': 'iso_code'})
        power_df = power_df[['unit_id', 'effectiveness', 'for_model',  'generation_type_id', 'iso_code']]
        power_df = pd.merge(power_df, countries,how='outer', on='iso_code', indicator=True)
        power_df = power_df[power_df['_merge'] == 'both']
        power_df = power_df[['unit_id', 'effectiveness', 'for_model',   'generation_type_id', 'iso_code', 'id']]
        power_df['for_model'].fillna(0, inplace=True)
        logger.info('got power data')
        return power_df


    def get_first_last_forecast_date(self):
        # получаем дату начала и конца моделирования из таблицы im.im_iteration
        dates_df = self.get_tables('im', 'im_iteration order by iteration_id')[['start_forecast_utc', 'end_forecast_utc']]
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
        max_year = date_df.iloc[-1].date()
        last_capacity = series_values['value'].iloc[-1]
        max_year = datetime.strptime(f"{max_year}", '%Y-%m-%d').year
        first_forecast_date, last_forecast_date = self.get_first_last_forecast_date()
        if max_year <= last_forecast_date.year + 1:
            d = []
            cap = []
            for date in range(max_year, last_forecast_date.year + 1):
                for month in range(1,13):
                    d.append(datetime.strptime(f"{date}-{month}-01", '%Y-%m-%d'))
                    cap.append(last_capacity)
            additional_series_values = pd.DataFrame(data={
                'series_id': [series_values['series_id'].values[0]] * 12 * (last_forecast_date.year + 1 - max_year),
                'frequency_id': [4] * 12 *(last_forecast_date.year + 1 - max_year),
                'd_date': d,
                'value': cap
            })
            series_values = pd.concat([series_values, additional_series_values])
        min_year = date_df.iloc[0].date()
        first_capacity = series_values['value'].iloc[0]
        min_year = datetime.strptime(f"{min_year}", '%Y-%m-%d').year
        if min_year >= first_forecast_date.year - 1:
            d = []
            cap = []
            for date in range(first_forecast_date.year, min_year):
                for month in range(1, 13):
                    d.append(datetime.strptime(f"{date}-{month}-01", '%Y-%m-%d'))
                cap = [first_capacity]*len(d)
            additional_series_values = pd.DataFrame(data={
                'series_id': [series_values['series_id'].values[0]] * len(d),
                'frequency_id': [4] * len(d),
                'd_date': d,
                'value': cap
            })
            series_values = pd.concat([series_values, additional_series_values], ignore_index=True)
            series_values.drop_duplicates(subset=['series_id', 'd_date'], keep = 'first')
            series_values['d_date'] = pd.to_datetime(series_values['d_date'])
            series_values.sort_values(by='d_date',ascending=True )
        return series_values



    def lignite(self):
        logger.info('running lignite')
        power_df = self.get_powerunits_country().query('generation_type_id == 7')

        lignite_cost = 1.76
        # series_values = self.get_tables('bi', 'series_data')
        # powerunit_df = self.get_tables('im', 'power_powerunit_info')
        lignite_coef = {167: 0.4069, 168: 0.4096, 169: 0.4096, 172: 0.4128, 174: 0.3429,
                        177: 0.3439, 180: 0.4802, 193: 1, 195:1 , 194:1, 197:1, 196: 1,
                        189: 0.3469, 192:0.4303, 203: 0.3734, 198: 0.6620}
        tickers = lignite_coef.keys()
        market_ticker = pd.DataFrame(data={
            'm_id': [21, 21, 21, 13, 24, 6, 23, 33, 34, 56, 35, 36, 25, 3, 11, 12],
            'series_id': [167, 168, 169, 172, 174, 177, 180, 193, 194, 197, 195, 196, 189, 192, 203, 198]

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
            one_ticker_df=pd.DataFrame()
            one_ticker_df['datetime'] = series_values.query('series_id ==@t')['d_date']
            one_ticker_df['value'] = series_values.query('series_id ==@t')['value']
            one_ticker_df['coef'] =[lignite_coef[t]]* len(series_values.query('series_id ==@t')['value'])
            one_ticker_df['gfc_val2'] = series_values.query('series_id ==@t')['value'] * lignite_coef[t]
            one_ticker_df['series_id'] = [t] * len(one_ticker_df['value'])
            one_ticker_df['m_id'] = series_values.query('series_id ==@t')['m_id']
            # one_ticker_df['iso_code'] = series_values.query('series_id ==@t')['iso_code']
            result_df=pd.concat([result_df, one_ticker_df])
        total_powerunits = pd.DataFrame()
        for country in set(market_ticker['m_id'].values): #цикл по странам
            power_country_df=power_df.query('id == @country')
            value_country_df = result_df.query('m_id == @country')
            if power_country_df.empty == False:
                for powerunit in set(power_country_df['unit_id'].values):#цикл по энергоблокам в стране
                    one_power_df=pd.DataFrame()
                    one_power_df['datetime'] = value_country_df['datetime']
                    one_power_df['powerunit_id'] = [powerunit]*len(value_country_df['datetime'])
                    one_power_df['country'] = [country]*len(value_country_df['datetime'])
                    one_power_df['effectiveness'] = [power_country_df.query('unit_id == @powerunit')['effectiveness'].values[0]]*len(value_country_df['datetime'])
                    one_power_df['series_id'] = value_country_df['series_id']
                    one_power_df['value'] = value_country_df['value']
                    one_power_df['gfc_val2'] = value_country_df['gfc_val2']
                    one_power_df['gfc_val3'] = one_power_df['gfc_val2']*one_power_df['effectiveness']
                    one_power_df['iso_code'] = [power_country_df['iso_code'].values[0]]*len(value_country_df['datetime'])
                    total_powerunits = pd.concat([total_powerunits, one_power_df])
        logger.info('lignite was done successfully')
        return total_powerunits, lignite_cost


    def coal(self, scenario):
        logger.info(f'running coal with {scenario} scenario')
        coal_cost = 1.76
        power_df = self.get_powerunits_country().query('generation_type_id == 6')

        powerunit_df = self.get_tables('im', 'power_powerunit_info')
        coal_coef_base = {165: 0.1639, 170: 0.1434, 173:0.1434, 176:0.1434,
                          179: 0.1565, 187:0.1434, 190:0.1434, 201:0.1434,
                          206:0.1434, 199:0.1434, 204:0.1434, 208:0.1434,
                          185:0.1434, 245:0.1434, 244:0.1434, 246:0.1434
        }
        coal_coef_low = {166:0.1639, 171:0.1434, 175:0.1434, 178:0.1434,
                         181: 0.1565, 188:0.1434, 191:0.1434, 202:0.1434,
                         207:0.1434, 200:0.1434, 205:0.1434, 209:0.1434, 186:0.1434}

        coal_coef_high = {210:0.1639, 211:0.1434, 212:0.1434, 213:0.1434,
                          214: 0.1565, 216:0.1434, 217:0.1434, 219:0.1434,
                          221:0.1434, 218:0.1434, 220:0.1434, 222:0.1434, 215:0.1434}
        market_ticker = pd.DataFrame(data={
            'm_id': [21, 21, 21, 13,13, 13, 24, 24, 24, 6, 6, 6, 23, 23, 23, 25, 25, 25, 3, 3, 3, 11, 11, 11,
                     26, 26, 26, 15, 15, 15, 20, 20, 20, 22, 22, 22, 4, 4, 4, 7, 10, 55],
            'series_id': [165, 210, 166, 170, 211, 171, 173, 212, 175, 176, 213, 178, 179, 214, 181, 187, 216, 188,
                          190, 217, 191, 201, 219, 202, 206, 221, 207, 199, 218, 200, 204, 220, 205, 208, 222, 209,
                          185, 215, 186, 245, 244, 246]})
        if scenario == 1:
            coal_coef = coal_coef_base
        elif scenario ==2:
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
            print(t)
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
            if power_country_df.empty == False:
                for powerunit in set(power_country_df['unit_id'].values):  # цикл по энергоблокам в стране
                    one_power_df = pd.DataFrame()
                    one_power_df['datetime'] = value_country_df['datetime']
                    one_power_df['powerunit_id'] = [powerunit] * len(value_country_df['datetime'])
                    one_power_df['country'] = [country] * len(value_country_df['datetime'])
                    one_power_df['series_id'] = value_country_df['series_id']
                    one_power_df['value'] = value_country_df['value']
                    one_power_df['effectiveness'] = [power_country_df.query('unit_id == @powerunit')[
                                                         'effectiveness'].values[0]] * len(value_country_df['datetime'])

                    one_power_df['gfc_val2'] = value_country_df['gfc_val2']
                    one_power_df['gfc_val3'] = one_power_df['gfc_val2'] * one_power_df['effectiveness']
                    one_power_df['iso_code'] = [power_country_df['iso_code'].values[0]] * len(
                        value_country_df['datetime'])
                    # print(one_power_df)
                    total_powerunits = pd.concat([total_powerunits, one_power_df])
        logger.info('coal was done successfully')
        return total_powerunits, coal_cost

    def gas(self, scenario):
        logger.info(f'running gas with {scenario} scenario')
        gas_cost = 1.8436
        power_df = self.get_powerunits_country().query('generation_type_id == 2')
        m_id = [21, 13, 6, 23, 25, 3, 4, 33, 34, 56, 35, 36, 12, 41, 20, 7, 37, 27, 31, 24, 11, 26, 22, 15, 10, 55, 2]
        if scenario == 1 :
            market_ticker = pd.DataFrame(data={
                'm_id': m_id,
                'series_id': [183] * len(m_id)})

        elif scenario == 2:
            market_ticker = pd.DataFrame(data={
                'm_id': m_id,
                'series_id': [225]*len(m_id)})
            # print(market_ticker)
        else:
            market_ticker = pd.DataFrame(data={
                'm_id': m_id,
                'series_id': [224] * len(m_id)})

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
            one_ticker_df['gfc_val2'] = series_values.query('series_id ==@t')['value'] +0.3
            one_ticker_df['series_id'] = [t] * len(one_ticker_df['value'])
            one_ticker_df['m_id'] = series_values.query('series_id ==@t')['m_id']
            result_df = pd.concat([result_df, one_ticker_df])
        total_powerunits = pd.DataFrame()
        for country in set(market_ticker['m_id'].values):  # цикл по странам
            power_country_df = power_df.query('id == @country')
            value_country_df = result_df.query('m_id == @country')
            if power_country_df.empty == False:
                for powerunit in set(power_country_df['unit_id'].values):  # цикл по энергоблокам в стране
                    one_power_df = pd.DataFrame()
                    one_power_df['datetime'] = value_country_df['datetime']
                    one_power_df['powerunit_id'] = [powerunit] * len(value_country_df['datetime'])
                    one_power_df['country'] = [country] * len(value_country_df['datetime'])
                    one_power_df['series_id'] = value_country_df['series_id']
                    one_power_df['value'] = value_country_df['value']
                    one_power_df['effectiveness'] = [power_country_df.query('unit_id == @powerunit')[
                                                         'effectiveness'].values[0]] * len(value_country_df['datetime'])

                    one_power_df['gfc_val2'] = value_country_df['gfc_val2']
                    one_power_df['gfc_val3'] = one_power_df['gfc_val2'] * one_power_df['effectiveness']
                    one_power_df['iso_code'] = [power_country_df['iso_code'].values[0]] * len(
                        value_country_df['datetime'])
                    # print(one_power_df)
                    total_powerunits = pd.concat([total_powerunits, one_power_df])
                    logger.info('gas was done successfully')
        return total_powerunits, gas_cost

    def co2_for_past_periods(self): #242-31

        m_id = [21,13, 24, 6, 23, 4, 10, 25, 3, 11, 12, 37, 20, 2, 15, 26, 22,7 , 27]
        market_ticker = pd.DataFrame(data={
            'm_id': m_id,
            'series_id': [182] * len(m_id)})
        market_ticker_new = pd.DataFrame(data = {'m_id':[31], 'series_id':[242]})
        market_ticker = pd.concat([market_ticker, market_ticker_new], ignore_index=True)
        # print(market_ticker)
        tickers = set(market_ticker['series_id'].values)
        series_values = pd.DataFrame()
        for t in tickers:
            series_value = self.get_the_furthest_year_of_product(t)
            series_values = pd.concat([series_values, series_value], ignore_index=True)
        date = datetime.now()
        series_values = series_values[series_values['d_date']<= date]
        past_co2_df = self.co2(series_values, market_ticker)
        past_co2_df = past_co2_df[past_co2_df['_merge'] == 'both']
        past_co2_df.fillna(0, inplace = True)
        return past_co2_df


    def co2_for_future_periods(self): #242-31
        m_id = [21,13, 24, 6, 23, 4, 10, 25, 3, 11, 12, 37, 20, 2, 15, 26, 22,7 , 27]
        market_ticker = pd.DataFrame(data={
            'm_id': m_id,
            'series_id': [182] * len(m_id)})
        market_ticker_new = pd.DataFrame(data = {'m_id':[31], 'series_id':[242]})
        market_ticker = pd.concat([market_ticker, market_ticker_new], ignore_index=True)
        # print(market_ticker)
        tickers = set(market_ticker['series_id'].values)
        series_values = pd.DataFrame()
        for t in tickers:
            series_value = self.get_the_furthest_year_of_product(t)
            series_values = pd.concat([series_values, series_value], ignore_index=True)
        date = datetime.now()
        series_values = series_values[series_values['d_date']>= date]
        future_co2_df = self.co2(series_values, market_ticker)
        future_co2_df = future_co2_df[future_co2_df['_merge'] == 'both']
        future_co2_df.fillna(0, inplace=True)
        return future_co2_df


    def co2(self, series_values, market_ticker):
        series_values = pd.merge(series_values, market_ticker, how='outer', on='series_id', indicator=True)
        series_values = series_values[series_values['_merge'] == 'both']
        series_values.drop(columns='_merge')
        result_df = pd.DataFrame()
        countries = set(market_ticker['m_id'].values)

        for t in countries:
            power_df = self.get_powerunits_country()
            power_df = power_df.rename(columns={'id': 'm_id'})
            # print(t)
            one_ticker_df = pd.DataFrame()
            one_ticker_df['series_id'] = series_values.query('m_id ==@t')['series_id']
            serie_id = one_ticker_df['series_id'].values[0]
            one_ticker_df['datetime'] = series_values.query('series_id ==@serie_id')['d_date']
            one_ticker_df['m_id'] = [t] * len(one_ticker_df['datetime'])
            power_df = power_df[power_df['m_id'] == t]
            if power_df.empty == False:
                power_df['for_model'].fillna(0, inplace=True)
                one_ticker_df = pd.merge(one_ticker_df, power_df[['unit_id', 'm_id', 'for_model']], how='outer', on='m_id',
                                         indicator=True)
                one_ticker_df = one_ticker_df[one_ticker_df['_merge'] == 'both']
            else:
                one_ticker_df['for_model'] = [0] * len(one_ticker_df['datetime'])

            one_ticker_df['value'] = series_values.query('series_id ==@serie_id')['value']
            one_ticker_df['gfc_val5'] = series_values.query('series_id ==@serie_id')['value'] \
                                        * 0.98 * one_ticker_df['for_model']
            result_df = pd.concat([result_df, one_ticker_df], ignore_index=True)
            result_df = result_df[result_df.value.notnull()]

        return result_df

    def get_last_iteration(self):
        iteration_df = self.get_tables('im', 'im_iteration')
        iteration = max(iteration_df['iteration_id'].values)
        logger.info(f'got last iteration {iteration}')
        return iteration


    def get_df_per_hour(self,commodity_df, commodity_cost, scenario):
        logger.info('starting to prepare main data to insert into table')
        powerunits = set(commodity_df['powerunit_id'].values)
        co2_df_past = self.co2_for_past_periods()
        co2_df_future = self.co2_for_future_periods()
        co2_df = pd.concat([co2_df_past, co2_df_future], ignore_index=True)
        for powerunit in powerunits:
            logger.info(f"collecting data from {powerunit} powerunit")
            # co2_df = co2_df.query('unit_id == @powerunit')
            commodity_df = commodity_df.query('powerunit_id == @powerunit')
            dates = sorted(pd.to_datetime(commodity_df['datetime']))
            co2_df = pd.concat([co2_df_past, co2_df_future], ignore_index=True)
            for d in range(0,len(dates)-1):

                da = [dates[d]]

                delta = dates[d+1] - dates[d]
                delta = int(f"{delta}".split(' ')[0]) * 24
                # print(delta)
                for t in range(1,delta):
                    da.append(da[t-1] + timedelta(hours=1))

                # print(main_df)
                # date = main_df['datetime_local'].values[0]
                x = dates[d]
                gfc_val2 = commodity_df.query("powerunit_id == @powerunit")\
                [commodity_df['datetime'] == x]['gfc_val2'].values[0]
                gfc_val3 = commodity_df.query("powerunit_id == @powerunit") \
                    [commodity_df['datetime'] == x]['gfc_val3'].values[0]
                country = commodity_df.query("powerunit_id == @powerunit") \
                    [commodity_df['datetime'] == x]['country'].values[0]
                # print(co2_df)
                gfc_val4_df = co2_df.query('unit_id == @powerunit') \
                [co2_df['datetime'] == x]['value']
                gfc_val5_df = co2_df.query('unit_id == @powerunit') \
                    [co2_df['datetime'] == x]['gfc_val5']
                if gfc_val4_df.empty == False:
                    gfc_val4 = gfc_val4_df.values[0]
                else:
                    gfc_val4 = 0

                if gfc_val5_df.empty == False:
                    gfc_val5 = gfc_val5_df.values[0]
                else:
                    gfc_val5 = 0
                gfc_val6 = None
                gfc_val7 = commodity_cost
                gfc_val8 = None
                gfc_val1 = gfc_val3 + gfc_val5 + gfc_val7
                main_df = pd.DataFrame()
                main_df['gfc_iteration'] = [self.get_last_iteration()]*len(da)
                main_df['gfc_scenario'] = [scenario]*len(da)
                main_df['gfc_local_datetime'] = da
                main_df['gfc_utc_datetime'] = substract_time_shift(da,
                                      entsoe_configs.COUTRIES_SHIFTS[f"{commodity_df['iso_code'].values[0]}"][1],
                                      entsoe_configs.COUTRIES_SHIFTS[f"{commodity_df['iso_code'].values[0]}"][0])

                main_df['gfc_market_id'] = [country]*len(da)
                main_df['gfc_generationunit_id'] = [powerunit] * len(da)
                main_df['gfc_microservice_id'] = [1] * len(da)
                main_df['gfc_indicator_id'] = [14] * len(da)
                main_df['gfc_val_1'] = [gfc_val1] * len(da)
                main_df['gfc_val_2'] = [ gfc_val2]* len(da)
                main_df['gfc_val_3'] = [gfc_val3] * len(da)
                main_df['gfc_val_4'] = [gfc_val4] * len(da)
                main_df['gfc_val_5'] = [gfc_val5] * len(da)
                main_df['gfc_val_6'] = [gfc_val6] * len(da)
                main_df['gfc_val_7'] = [gfc_val7] * len(da)
                main_df['gfc_val_8'] = [gfc_val8] * len(da)
                main_df.drop_duplicates(subset=['gfc_iteration' ,'gfc_utc_datetime', 'gfc_scenario', 'gfc_generationunit_id', 'gfc_microservice_id'])
                logger.info('duplicates were dropped')
                insert_into_table(main_df, 'im', 'im_generationunit_forecast_calc',
                                               custom_conflict_resolution='on conflict do nothing')
                # main_df.to_sql('im_generationunit_forecast_calc', con=envs('ALCHEMY_CONNECTION', cast=str), schema='im',
                #                if_exists='append', index=False)
                logger.info('data was inserted into im_generationunit_forecast_calc table')
                # print(main_df)

    def run_commodities(self, scenario):
        lignite_df, lignite_cost = self.lignite()
        coal_df, coal_cost = self.coal(scenario)
        gas_df, gas_cost = self.gas(scenario)
        self.get_df_per_hour(lignite_df, lignite_cost,scenario)
        self.get_df_per_hour(coal_df, coal_cost, scenario)
        self.get_df_per_hour(gas_df, gas_cost, scenario)


    def run_all_scenario(self):
        scenario_list = [1,2,3,4]
        for scenario in scenario_list:
            logger.info(f'run scenario {scenario}')
            self.run_commodities(scenario)