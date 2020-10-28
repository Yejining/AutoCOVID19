import requests
import json
import pandas as pd

KEY = 'AIzaSyCeRYD2V3a2R6AO_OcHp7Ar0b2bzhE8NUM'


class GeoCoder:
    def __init__(self, key):
        self.key = key

    def append_seoul(self, keyword):
        return '서울 %s' % keyword

    def generate_url(self, keyword):
        base_url = 'https://maps.googleapis.com/maps/api/geocode/json?address='
        url = '%s%s&key=%s' % (base_url, self.append_seoul(keyword), self.key)
        url = url.replace(' ', '%20')

        return url

    def request(self, url):
        response = requests.get(url)
        response_text = response.text
        response_dict = json.loads(response_text)
        return response_dict

    def get_address(self, keyword):
        url = self.generate_url(keyword)
        address = self.request(url)
        return address

    def extract_info(self, result):
        geometry = result['geometry']
        location = geometry['location']
        latitude = location['lat']
        longitude = location['lng']
        if len(result['types']) == 0:
            place_type = ''
        else:
            if result['types'][0] == 'establishment' or result['types'][0] == 'political' or \
                result['types'][0] == 'street_address' or result['types'][0] == 'accounting' or \
                result['types'][0] == 'subpremise' or result['types'][0] == 'locality':
                if len(result['types']) == 1:
                    place_type = result['types'][0]
                else:
                    place_type = result['types'][1]
            else:
                place_type = result['types'][0]

        info = {'lat': latitude, 'lng': longitude, 'type': place_type}
        return info

    def get_information(self, keyword):
        results = self.get_address(keyword)
        results = results['results']

        information = pd.DataFrame()
        for result in results:
            info = self.extract_info(result)
            information = information.append(info, ignore_index=True)
            break

        return information


if __name__ == '__main__':
    geo_api = GeoCoder(key='AIzaSyCeRYD2V3a2R6AO_OcHp7Ar0b2bzhE8NUM')

    information = geo_api.get_information('성북 동작대로27길 16-6 쏘맥의 달인')

    print(information)
