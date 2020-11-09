import pandas as pd
import requests
import os

from urllib.parse import urlencode, quote_plus
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from pathlib import Path
from os.path import join


KEY = ''


class StatusAPI:
    def __init__(self):
        self.key = KEY

    def request(self, start_date, end_date):
        start_date = datetime.strptime(start_date, '%Y%m%d')
        end_date = datetime.strptime(end_date, '%Y%m%d')
        delta = end_date - start_date

        result_df = pd.DataFrame()
        for i in range(delta.days + 1):
            new_date = start_date + timedelta(days=i)
            str_date = new_date.strftime('%Y%m%d')

            items = self._request(str_date, str_date)
            item_df = self._items_to_df(items, str_date)
            result_df = result_df.append(item_df, ignore_index=True)

        result_df = result_df.sort_values('data_date')
        result_df = result_df.reset_index(drop=True)

        result_df['new_pat'] = result_df['new_pat'].astype(int)
        result_df['no_pat'] = result_df['no_pat'].astype(int)

        new_columns = ['District', 'State', 'data_date', 'new_pat', 'pop', 'long', 'lat', 'no_pat']
        result_df = result_df[new_columns]
        return result_df

    def _request(self, start_date, end_date):
        base_url = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19SidoInfStateJson'
        base_url += '?serviceKey=%s&' % self.key
        query_params = urlencode({quote_plus('pageNo'): '1',
                                  quote_plus('numOfRows'): '1',
                                  quote_plus('startCreateDt'): start_date,
                                  quote_plus('endCreateDt'): end_date})
        url = '%s%s' % (base_url, query_params)

        response = requests.get(url)
        response.encoding = None
        response_text = response.text

        soup = BeautifulSoup(response_text, 'html.parser')
        body = soup.find('items')
        items = body.find_all('item')

        return items

    def _items_to_df(self, items, str_date=None):
        result_df = pd.DataFrame()

        if len(items) == 0:
            new_dict = {'District': 'Seoul', 'State': 'Seoul',
                        'pop': 9689159, 'lat': 37.5665, 'long': 126.978}
            new_dict.update({'data_date': str_date})
            new_dict.update({'new_pat': -1})
            new_dict.update({'no_pat': -1})

            result_df = result_df.append(new_dict, ignore_index=True)
            return result_df

        for item in items:
            region = item.find('gubun').text  # 지역
            if region != '서울': continue

            new_dict = {'District': 'Seoul', 'State': 'Seoul',
                        'pop': 9689159, 'lat': 37.5665, 'long': 126.978}

            std_day = item.find('stdday')
            std_day = -1 if std_day is None else std_day.text  # 확진자 수
            std_day = datetime.strptime(std_day, '%Y년 %m월 %d일 %H시')
            std_day = std_day.strftime('%Y%m%d')
            new_dict.update({'data_date': std_day})

            inc_dec = item.find('incdec')
            inc_dec = -1 if inc_dec is None else int(inc_dec.text)  # 전일대비 증감수
            new_dict.update({'new_pat': inc_dec})

            def_cnt = item.find('defcnt')
            def_cnt = -1 if def_cnt is None else int(def_cnt.text)  # 확진자 수
            new_dict.update({'no_pat': def_cnt})

            result_df = result_df.append(new_dict, ignore_index=True)

        return result_df


if __name__ == '__main__':
    root = Path(os.getcwd())

    status_api = StatusAPI()
    result_df = status_api.request('20200122', '20200902')

    path = join(root, 'comparatives', 'covid19spatiotemporal', 'dataset')
    Path(path).mkdir(parents=True, exist_ok=True)
    print('saving api result into %s' % join(path, 'Korea_Covid_Patient.csv'))
    result_df.to_csv(join(path, 'Korea_Covid_Patient.csv'), encoding='utf-8-sig', index=False)
