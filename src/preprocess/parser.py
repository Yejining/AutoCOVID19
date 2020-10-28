from collections import defaultdict

import bs4
import requests
import pandas as pd
import os
import re

from bs4 import BeautifulSoup, NavigableString, Tag
from pathlib import Path


class Parser:
    def __init__(self, url, table_class):
        self.url = url
        self.table_class = table_class

    def request_status(self):
        result = requests.get(self.url)
        content = result.content
        return content

    def save_html(self, path, content):
        with open(path, 'wb') as f:
            f.write(content)

    def read_html(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        return text

    def extract_table(self, content):
        pass

    def extract_info(self, table):
        pass

    def save_info(self, path, info):
        parent = Path(path).parent
        Path(parent).mkdir(parents=True, exist_ok=True)

        if os.path.isfile(path):
            info.to_csv(path, encoding='utf-8-sig', mode='a', index=False, header=False)
        else:
            info.to_csv(path, encoding='utf-8-sig', mode='a', index=False)

    def parse_and_save(self, file_list, content_path, saving_path):
        for file in file_list:
            print(file)
            file_path = os.path.join(content_path, file)
            content = self.read_html(file_path)
            table = self.extract_table(content)
            patient_info = self.extract_info(table)
            self.save_info(saving_path, patient_info)

    def get_between_br_tags(self, rows):
        text_list = []

        for row in rows:
            for br in row.findAll('br'):
                next_s = br.nextSibling
                if not (next_s and isinstance(next_s, NavigableString)):
                    continue
                next2_s = next_s.nextSibling
                if next2_s and isinstance(next2_s, Tag) and next2_s.name == 'br':
                    text = str(next_s).strip()
                    if text:
                        text_list.append(text)

        return text_list


class StatusParser(Parser):
    def __init__(self, url, table_class):
        super(StatusParser, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', attrs={'class': self.table_class})
        return table

    def extract_info(self, table):
        patients = pd.DataFrame()
        table_body = table.tbody

        rows = table_body.find_all(['tr'])
        for row in rows:
            elements = row.find_all(['td'])
            elem_dict = dict()
            for elem in elements:
                name = elem['data-tit']
                value = elem.text
                elem_dict.update({name: value})
            patients = patients.append(elem_dict, ignore_index=True)

        patients['확진일'] = '2020.' + patients['확진일'].astype(str)
        patients['확진일'] = pd.to_datetime(patients['확진일'], format='%Y.%m.%d.')

        return patients


class GangnamParser1(Parser):
    def __init__(self, url, table_class):
        super(GangnamParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('div', attrs={'id': self.table_class})
        table = table.find('div', attrs={'class': 'pathAcod'})
        return table

    def extract_info(self, table):
        patients = pd.DataFrame()
        route_all = table.find_all('div', attrs={'class': 'acodList'})

        for index, patient_route in enumerate(route_all):
            if index == len(route_all) - 1: break
            elem_dict = dict()
            patient_id = patient_route.find('div', attrs={'class': 'pathTitle'})
            elem_dict.update({'id': patient_id.text})

            days = patient_route.find_all('div', attrs={'class': 'mb10'})
            for day in days:
                date = day.find('h5').text
                elem_dict.update({'date': date})

                routes_ul = day.find('ul', attrs={'class': 'ml20'})
                routes = routes_ul.find_all('li')
                for route in routes:
                    span = route.find('span')
                    span = span.text if span is not None else ''
                    route_text = route.text.replace(span, '')
                    real_dict = elem_dict.copy()
                    real_dict.update({'route': route_text})
                    patients = patients.append(real_dict, ignore_index=True)

        return patients


class GangnamParser2(GangnamParser1):
    def __init__(self, url, table_class):
        super(GangnamParser2, self).__init__(url, table_class)

    def extract_info(self, table):
        patients = pd.DataFrame()
        route_all = table.find_all('div', attrs={'class': 'acodList'})

        for index, patient_route in enumerate(route_all):
            patient_id = patient_route.find('div', attrs={'class': 'pathTitle'})

            route = patient_route.find('tbody')
            if route is None: continue
            places = route.find_all('tr')
            for place in places:
                elem_dict = dict()
                elem_dict.update({'id': patient_id.text})

                elements = place.find_all('td')
                titles = ['type', 'region', 'location', 'date', 'disinfection']
                for index2, elem in enumerate(elements):
                    if index2 == len(titles): index2 -= 1
                    elem_dict.update({titles[index2]: elem.text})
                patients = patients.append(elem_dict, ignore_index=True)
            break

        return patients


class GangdongParser1(Parser):
    def __init__(self, url, table_class):
        super(GangdongParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', attrs={'class': self.table_class})
        table = table.find('tbody')
        return table

    def extract_info(self, table):
        patients = pd.DataFrame()

        days = table.find_all('tr')
        elem_dict = dict()
        for day in days:
            elements = day.find_all('td')
            if len(elements) == 3 or len(elements) == 4: continue
            indexer = 3 if len(elements) > 2 else 0
            if len(elements) > 2:
                local_id = elements[0].find('b')
                global_id = re.search(r'<b>(.*?)<br\/>(.*?)(<\/b>)', str(local_id)).group(1)
                local_id = re.search(r'<b>(.*?)<br\/>(.*?)(<\/b>)', str(local_id)).group(2)
                confirmed_date = elements[1].text
                personal_info = elements[2].text
                elem_dict.update({'local_id': local_id})
                elem_dict.update({'global_id': global_id})
                elem_dict.update({'confirmed_date': confirmed_date})
                elem_dict.update({'personal_info': personal_info})
            date = elements[indexer + 0].text
            elem_dict.update({'date': date})
            routes = elements[indexer + 1].find_all('p')
            for route in routes:
                regex1 = r'<p>(<strong>(.*?)<\/strong>)*(.*?)(<br\/)*(.*?)(<\/p>)'
                route_edit1 = re.search(regex1, str(route)).group(5)
                if len(route_edit1) == 0: continue

                regex2 = r'(.*)(<br\/>)'
                route_edit2 = re.search(regex2, route_edit1)
                place = route_edit1 if route_edit2 is None else route_edit2.group(1)
                elem_dict.update({'route': place})
                patients = patients.append(elem_dict, ignore_index=True)

        return patients


class GangdongParser2(Parser):
    def __init__(self, url, table_class):
        super(GangdongParser2, self).__init__(url, table_class)

    def pass_ids(self, ids):
        self.ids = ids

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        tables = soup.find_all('table', attrs={'class': self.table_class})
        return tables

    def extract_info(self, tables):
        titles = ['region', 'type', 'route', 'date', 'disinfection']
        patients = pd.DataFrame()
        index = 0

        for table in tables:
            body = table.find('tbody')
            rows = body.find_all('tr')
            if len(rows) < 2: continue

            patient = pd.DataFrame()
            for index1, row in enumerate(rows):
                if index1 == 0: continue
                row_dict = dict()
                elements_td = row.find_all('td')
                for index2, elem in enumerate(elements_td):
                    row_dict.update({titles[index2]: elem.text})
                patient = patient.append(row_dict, ignore_index=True)

            patient['id'] = self.ids[self.index][index]
            patient['address'] = '강동구'
            index += 1

            patients = pd.concat([patients, patient])

        return patients

    def parse_and_save(self, file_list, content_path, saving_path):
        for index, file in enumerate(file_list):
            print(file)
            self.index = index
            file_path = os.path.join(content_path, file)
            content = self.read_html(file_path)
            table = self.extract_table(content)
            patient_info = self.extract_info(table)
            self.save_info(saving_path, patient_info)


class GangbukParser1(Parser):
    def __init__(self, url, table_class):
        super(GangbukParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        content_div = soup.find('div', attrs={'class': self.table_class})
        li_list = content_div.find_all('li', attrs={'class': 'item'})

        return li_list

    def extract_info(self, li_list):
        patients = pd.DataFrame()

        for li in li_list:
            sub_dict = dict()
            sub_li_list = li.find_all('li')
            for sub_li in sub_li_list:
                title = sub_li.find('em').text.replace(':', '')
                value = sub_li.find('p').text
                sub_dict.update({title: value})
            route_div = li.find('div', attrs={'class': 'layer'})
            routes = route_div.find_all('p')
            for route in routes:
                route_dict = sub_dict.copy()
                route_split = route.text.split('|')
                route_dict.update({'date': route_split[0]})
                route_rest = ''
                for i in range(1, len(route_split)):
                    route_rest += route_split[i]
                route_dict.update({'route': route_rest})
                patients = patients.append(route_dict, ignore_index=True)

        cols = ['인적사항', 'date', '확진일', '감염경로', '접촉', 'route', '격리시설']
        patients = patients[cols]
        return patients


class GangseoParser1(Parser):
    def __init__(self, url, table_class):
        super(GangseoParser1, self).__init__(url, table_class)

    def read_html(self, path):
        with open(path, 'r', encoding='cp949') as f:
            text = f.read()

        return text

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', attrs={'class': self.table_class})
        return table

    def extract_info(self, table):
        personal_infos = table.find_all('tr', attrs={'class': 'folder'})
        personal_routes = table.find_all('tr', attrs={'class': 'fold'})

        patients = pd.DataFrame()
        for index, personal_info in enumerate(personal_infos):
            personal_route = personal_routes[index].find('td')

            patient_dict = dict()
            cols_td = personal_info.find_all('td')
            id = re.search(r'<strong>(.*)<\/strong>', str(cols_td[0])).group(1)
            patient_dict.update({'id': id})
            patient_dict.update({'개인정보': cols_td[1].text})
            patient_dict.update({'confirmed_date': cols_td[2].text})

            remove_first_tag = re.match(r'<td colspan="4">(.*)<br\/>', str(personal_route))

            reason = re.match(r'^\[(.*)\]<br\/>', remove_first_tag.group(1))
            if reason is None: continue
            reason = reason.group(1)
            patient_dict.update({'reason': reason})

            for br in personal_route.findAll('br'):
                next_s = br.nextSibling
                if not (next_s and isinstance(next_s, NavigableString)):
                    continue
                next2_s = next_s.nextSibling
                if next2_s and isinstance(next2_s, Tag) and next2_s.name == 'br':
                    text = str(next_s).strip()
                    if text:
                        if next_s[0] == '-':
                            date = re.search(r'- (.*)', next_s).group(1)
                            patient_dict.update({'date': date})
                        else:
                            route = re.search(r'\· (\d+:\d+)(\s*\~\s*\d+:\d+)* (.*)', next_s)
                            if route is None: continue
                            route = route.group(3)
                            patient_row = patient_dict.copy()
                            patient_row.update({'route': route})
                            patients = patients.append(patient_row, ignore_index=True)

        cols = ['id', '개인정보', 'confirmed_date', 'reason', 'date', 'route']
        patients = patients[cols]
        return patients


class GangseoParser2(GangseoParser1):
    def __init__(self, url, table_class):
        super(GangseoParser2, self).__init__(url, table_class)

    def extract_info(self, table):
        info_list = table.find_all('td', attrs={'class': 'contsview'})
        info_list = self.get_between_br_tags(info_list)

        isFilled = False
        patients = pd.DataFrame()
        patient_dict = dict()
        for info in info_list:
            if info[0] == '○':
                id = re.search(r'[가-힣]*\s\d*번', info).group(0)
                address = re.search(r'\((.*)\)', info).group(1)
                patient_dict.update({'patient': id})
                patient_dict.update({'address': address})
                isFilled = True
            elif isFilled is False:
                continue
            elif info[0] == '-':
                date = re.search(r'\d*\.\s*\d*\.\([가-힣]\)', info)
                if date is None: continue
                else: date = date.group(0)
                route = re.search(r'[가-힣]\)\s*(.*)', info).group(1)
                patient_dict.update({'date': date})
                if len(route) != 0:
                    patient_dict.update({'route': route})
                    patients = patients.append(patient_dict, ignore_index=True)
            elif info[0] == '·':
                patient_dict.update({'route': info})
                patients = patients.append(patient_dict, ignore_index=True)
            else:
                isFilled = False

        cols = ['patient', 'address', 'date', 'route']
        patients = patients[cols]
        return patients


class GangseoParser3(GangseoParser1):
    def __init__(self, url, table_class):
        super(GangseoParser3, self).__init__(url, table_class)

    def extract_info(self, table):
        info_list = table.find_all('td', attrs={'class': 'contsview'})
        info_list = self.get_between_br_tags(info_list)

        patients = pd.DataFrame()
        patient_dict = dict()
        for info in info_list:
            if info[0] == '○':
                patient_dict = dict()
                region = re.search(r'[가-힣]*구*', info)
                if region is None: continue
                ids = re.findall(r'\d*번', info)
                reason = re.search(r'\((.*)\)', info)
                if reason is None:
                    patient_dict.update({'reason': 'None'})
                else:
                    patient_dict.update({'reason': reason.group(1)})
                patient_dict.update({'address': region.group()})
                patient_dict.update({'ids': ids})
            elif info[0] == '-' or info[0] == '=':
                if len(patient_dict) == 0: continue
                date = re.search(r'\d*\.\s*\d*\.*\([가-힣]\)', info).group(0)
                route = re.search(r'[가-힣]\)\s*(.*)', info).group(1)
                patient_dict.update({'date': date})
                if len(route) != 0:
                    patient_dict.update({'route': route})
                    patients = patients.append(patient_dict, ignore_index=True)
            elif info[0] == '·':
                if len(patient_dict) == 0: continue
                patient_dict.update({'route': info})
                patients = patients.append(patient_dict, ignore_index=True)

        new_patients = pd.DataFrame()

        indices_to_del = []
        for index, row in patients.iterrows():
            ids = row['ids']
            if len(ids) == 0: continue
            if len(ids) > 1:
                indices_to_del.append(index)
                for id in ids:
                    new_row = row.copy().to_dict()
                    new_row.update({'ids': id})
                    new_patients = new_patients.append(new_row, ignore_index=True)
            else:
                new_row = row.copy().to_dict()
                new_row.update({'ids': row['ids'][0]})
                new_patients = new_patients.append(new_row, ignore_index=True)

        new_patients = new_patients.rename(columns={'ids': 'id'})

        cols = ['id', 'address', 'date', 'reason', 'route']
        new_patients = new_patients[cols]
        return new_patients


class GangseoParser4(GangseoParser1):
    def __init__(self, url, table_class):
        super(GangseoParser4, self).__init__(url, table_class)

    def extract_info(self, table):
        info_list = table.find_all('td', attrs={'class': 'contsview'})
        info_list = self.get_between_br_tags(info_list)

        patients = pd.DataFrame()
        patient_dict = dict()
        for info in info_list:
            if info[0] == '○':
                patient_dict = dict()
                region = re.search(r'○\s*[가-힣]*', info)
                if region is None: continue
                ids = re.findall(r'\d*번', info)
                patient_dict.update({'reason': 'None'})
                patient_dict.update({'address': region.group()[1:]})
                patient_dict.update({'ids': ids})
            elif info[0] == '▷':
                if len(patient_dict) == 0: continue
                date = re.search(r'\d*\.\s*\d*\.*\([가-힣]\)', info).group(0)
                route = re.search(r'[가-힣]\)\s*(.*)', info).group(1)
                patient_dict.update({'date': date})
                if len(route) != 0:
                    patient_dict.update({'route': route})
                    patients = patients.append(patient_dict, ignore_index=True)
            elif info[0] == '-' or info[0] == '·':
                if len(patient_dict) == 0: continue
                patient_dict.update({'route': info})
                patients = patients.append(patient_dict, ignore_index=True)

        new_patients = pd.DataFrame()

        indices_to_del = []
        for index, row in patients.iterrows():
            ids = row['ids']
            if len(ids) > 1:
                indices_to_del.append(index)
                for id in ids:
                    new_row = row.copy().to_dict()
                    new_row.update({'ids': id})
                    new_patients = new_patients.append(new_row, ignore_index=True)
            else:
                new_row = row.copy().to_dict()
                new_row.update({'ids': row['ids'][0]})
                new_patients = new_patients.append(new_row, ignore_index=True)

        new_patients = new_patients.rename(columns={'ids': 'id'})

        cols = ['id', 'address', 'date', 'reason', 'route']
        new_patients = new_patients[cols]
        return new_patients


class GangseoParser5(GangseoParser1):
    def __init__(self, url, table_class):
        super(GangseoParser5, self).__init__(url, table_class)

    def extract_info(self, table):
        info_list = table.find_all('td', attrs={'class': 'contsview'})
        info_list = self.get_between_br_tags(info_list)

        patients = pd.DataFrame()
        patient_dict = dict()
        for info in info_list:
            if info[0] == '※':
                patient_dict = dict()
                region = re.search(r'※\s*[가-힣]*', info)
                if region is None: continue
                ids = re.findall(r'\d*번', info)
                patient_dict.update({'reason': 'None'})
                patient_dict.update({'address': region.group()[1:]})
                patient_dict.update({'ids': ids})
            elif info[0] == '▷':
                if len(patient_dict) == 0: continue
                date = re.search(r'\d*\.\s*\d*\.*\([가-힣]\)', info).group(0)
                route = re.search(r'[가-힣]\)\s*(.*)', info).group(1)
                patient_dict.update({'date': date})
                if len(route) != 0:
                    patient_dict.update({'route': route})
                    patients = patients.append(patient_dict, ignore_index=True)
            elif info[0] == '-' or info[0] == '·':
                if len(patient_dict) == 0: continue
                patient_dict.update({'route': info})
                patients = patients.append(patient_dict, ignore_index=True)

        new_patients = pd.DataFrame()

        indices_to_del = []
        for index, row in patients.iterrows():
            ids = row['ids']
            if len(ids) > 1:
                indices_to_del.append(index)
                for id in ids:
                    new_row = row.copy().to_dict()
                    new_row.update({'ids': id})
                    new_patients = new_patients.append(new_row, ignore_index=True)
            else:
                new_row = row.copy().to_dict()
                new_row.update({'ids': row['ids'][0]})
                new_patients = new_patients.append(new_row, ignore_index=True)

        new_patients = new_patients.rename(columns={'ids': 'id'})

        cols = ['id', 'address', 'date', 'reason', 'route']
        new_patients = new_patients[cols]
        return new_patients


class GwanakParser1(Parser):
    def __init__(self, url, table_class):
        super(GwanakParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', attrs={'id': self.table_class})
        div = table.find('div', attrs={'class': 'se-main-container'})
        return div

    def extract_info(self, table):
        sub_tables = table.find_all('div', attrs={'class': 'se-component se-table se-l-default'})[1:]
        cols = ['확진자', '유형', '주소', '노출일자', '소독 여부']

        patients = pd.DataFrame()
        for sub_table in sub_tables:
            tbody = sub_table.find('tbody')
            tr_list = tbody.find_all('tr', attrs={'class': 'se-tr'})[1:]
            for tr in tr_list:
                contents = tr.find_all('td', attrs={'class': 'se-cell'})
                patient = dict()
                for index, content in enumerate(contents):
                    text = content.text.replace('\n', '')
                    patient.update({cols[index]: text})
                patients = patients.append(patient, ignore_index=True)

        patients = patients[cols]
        return patients

    def parse_and_save(self, file_list, content_path, saving_path):
        for file in file_list:
            print(file)
            file_path = os.path.join(content_path, '%s_files' % file, 'PostView.html')
            if os.path.isfile(file_path) is False:
                file_path = os.path.join(content_path, '%s_files' % file, 'PrologueList.html')
            content = self.read_html(file_path)
            table = self.extract_table(content)
            patient_info = self.extract_info(table)
            self.save_info(saving_path, patient_info)


class GuroParser1(Parser):
    def __init__(self, url, table_class):
        super(GuroParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table', attrs={'class': self.table_class})
        tbody = table.find('tbody')
        return tbody

    def extract_info(self, table):
        patient_infos = table.find_all('tr', attrs={'class': 'item moving'})
        patient_routes = table.find_all('tr', attrs={'class': 'hide'})

        patients = pd.DataFrame()

        for index, info in enumerate(patient_infos):
            patient = dict()
            td_list = info.find_all('td')
            if len(td_list) < 2: continue
            patient.update({'id': td_list[0].text})
            patient.update({'date': td_list[1].text})
            routes = patient_routes[index].find('td')
            routes_split = re.finditer('◯\s*([가-힣]|\s|\(|\)|·)*<br\/>', str(routes))
            for route in routes_split:
                new_patient = patient.copy()
                new_patient.update({'route': route.group(0)})
                patients = patients.append(new_patient, ignore_index=True)

        return patients


class NowonParser1(Parser):
    def __init__(self, url, table_class):
        super(NowonParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('div', attrs={'class': self.table_class})
        return table

    def extract_info(self, table):
        patient_infos = table.find_all('div', attrs={'class': 'accordion-title'})
        patient_routes = table.find_all('div', attrs={'class': 'accordion-content'})

        patients = pd.DataFrame()

        for index, info in enumerate(patient_infos):
            id = info.find('p', attrs={'class': 'cal-title'}).text
            if re.search(r'역학조사중', id) is not None: continue

            patient = dict()
            patient.update({'id': id})
            patient.update({'confirmed_date': info.find('p', attrs={'class': 'cal-day'}).text})

            routes = patient_routes[index].find_all('p')
            dict_date = ''
            for route in routes:
                date = re.search(r'\d*.\d*\([가-힣]\)', route.text)
                if date is not None:
                    dict_date = date.group()
                route_string = route.text
                new_patient = patient.copy()
                new_patient.update({'date': dict_date, 'route': route_string})
                patients = patients.append(new_patient, ignore_index=True)

        return patients


class DobongParser1(Parser):
    def __init__(self, url, table_class):
        super(DobongParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('div', attrs={'class': self.table_class})
        return table

    def extract_info(self, table):
        table_rows = table.find_all('div', attrs={'class': 'corona_info'})

        patients = pd.DataFrame()

        for row in table_rows:
            info = row.find('ul', attrs={'class': 'tit s_tit'})
            routes = str(row.find('div', attrs={'class': 'info_detail'}))

            patient = dict()
            id = info.find('li', attrs={'class': 'cell01'}).text
            if re.search((r'역학조사중'), id) is not None: continue

            patient.update({'id': id})
            patient.update({'confirmed_date': info.find('li', attrs={'class': 'cell03'}).text})
            patient.update({'reason': info.find('li', attrs={'class': 'cell05'}).text})

            sub_patient = self.get_sub_dataframe(routes)
            sub_patient['id'] = patient['id']
            sub_patient['confirmed_date'] = patient['confirmed_date']
            sub_patient['reason'] = patient['reason']

            patients = patients.append(sub_patient, ignore_index=True)

        return patients

    def get_sub_dataframe(self, string):
        is_date_exist = False

        indices = [(m.start(0), m.end(0)) for m in re.finditer(r'<br\/>', string)]

        start = 0

        patients = pd.DataFrame()
        patient = dict()

        for index in indices:
            end = index[0]

            row = string[start:end].replace('<br/>', '')

            start = end

            if len(row) == 0:
                continue

            if row[0] == '○' and is_date_exist is True:
                patient = dict()

            if row[0] == '□':
                is_date_exist = False
                continue

            if row[0] == '○':
                date = re.sub(r'○\s*', '', row)
                patient.update({'date': date})
                is_date_exist = True

            if is_date_exist is False:
                continue

            if re.search(r'^(\s*-)', row) is not None:
                route = re.sub(r'\s*-\s*', '', row)
                new_patient = patient.copy()
                new_patient.update({'route': route})
                patients = patients.append(new_patient, ignore_index=True)

            start = end

        return patients


class DongdaemoonParser1(Parser):
    def __init__(self, url, table_class):
        super(DongdaemoonParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        div = soup.find('div', attrs={'class': self.table_class})
        table = div.find('tbody')

        return table

    def extract_info(self, table):
        patients = pd.DataFrame()

        tr_list = table.find_all('tr', attrs={'class': 'text-in'})
        for tr in tr_list:
            patient = dict()

            p_list = tr.find_all('p')
            p_string = p_list[3].text

            pattern = r'#\d*번 확진자'
            patient_id = re.search(pattern, p_string).group()
            patient.update({'patient_id': patient_id})

            pattern = r'(?<=\()([[가-힣]|\s|\d|])*(?=[,|\)])'
            region = re.search(pattern, p_string).group()
            patient.update({'region': region})

            pattern = r'(?<=는 )([~\d|\.|\(|[가-힣]|\)|\s|:|‘|’|#|–|-|])*(?=,)'
            reason = re.search(pattern, p_string).group()
            patient.update({'reason': reason})

            tbody = tr.find('tbody')
            if tbody is None: continue
            inner_tr_list = tbody.find_all('tr')[1:]
            for inner_tr in inner_tr_list:
                inner_td_list = inner_tr.find_all('td')
                if len(inner_td_list) < 6: continue
                location_region = inner_td_list[0].text
                location_region = location_region.replace('\n', '')
                patient.update({'location_region': location_region})
                location_type = inner_td_list[1].text
                location_type = location_type.replace('\n', '')
                patient.update({'location_type': location_type})
                location_address = inner_td_list[2].text
                location_address = location_address.replace('\n', '')
                patient.update({'location_address': location_address})
                date = inner_td_list[3].text
                date = date.replace('\n', '')
                patient.update({'date': date})

                patients = patients.append(patient, ignore_index=True)

        return patients


class DongdaemoonParser2(Parser):
    def __init__(self, url, table_class):
        super(DongdaemoonParser2, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        div = soup.find('div', attrs={'class': self.table_class})
        table = div.find('tbody')

        return table

    def extract_info(self, table):
        patients = pd.DataFrame()
        patient = dict()

        td = table.find('td', attrs={'class': 'onPage boardCont'})

        p_list = td.find_all('p')
        p_string = p_list[2].text

        pattern = r'#\d*번 확진자'
        patient_id = re.search(pattern, p_string).group()
        patient.update({'patient_id': patient_id})

        pattern = r'(?<=\()([[가-힣]|\s|\d|])*(?=[,|\)])'
        region = re.search(pattern, p_string).group()
        patient.update({'region': region})

        pattern = r'(?<=는 )([~\d|\.|\(|[가-힣]|\)|\s|:|‘|’|#|–|-|])*(?=,)'
        reason = re.search(pattern, p_string)
        if reason is not None:
            patient.update({'reason': reason.group()})
        else:
            patient.update({'reason': '감염원인 모름'})

        tbody = td.find('tbody')
        tr_list = tbody.find_all('tr')[1:]
        for inner_tr in tr_list:
            inner_td_list = inner_tr.find_all('td')
            if len(inner_td_list) < 6: continue
            location_region = inner_td_list[0].text
            location_region = location_region.replace('\n', '')
            patient.update({'location_region': location_region})
            location_type = inner_td_list[1].text
            location_type = location_type.replace('\n', '')
            patient.update({'location_type': location_type})
            location_address = inner_td_list[2].text
            location_address = location_address.replace('\n', '')
            patient.update({'location_address': location_address})
            date = inner_td_list[3].text
            date = date.replace('\n', '')
            patient.update({'date': date})

            patients = patients.append(patient, ignore_index=True)

        return patients


class DongjakParser1(Parser):
    def __init__(self, url, table_class):
        super(DongjakParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        body = soup.find('body')
        ul = body.find('ul', attrs={'class': 'iw-arcordian'})
        return ul

    def extract_info(self, table):
        patients = pd.DataFrame()

        patient_info_list = table.find_all('a', attrs={'class': 'iw-ask'})
        patient_route_list = table.find_all('div', attrs={'class': 'iw-answer'})

        for index, patient_info in enumerate(patient_info_list):
            info_df = self.parse_patient_info(patient_info.text)
            if info_df is None: continue

            patient_routes = patient_route_list[index]
            patient_routes = patient_routes.find_all('p', attrs={'class': 'iwa-exp'})
            if len(patient_routes) == 0: continue
            else: patient_routes = patient_routes[1]
            route_df = self.parse_patient_route(str(patient_routes))

            for index2, row in info_df.iterrows():
                for col_name, col_data in row.iteritems():
                    route_df[col_name] = col_data

            patients = patients.append(route_df, ignore_index=True)

        cols = ['patient_id', 'region', 'confirmed_date', 'date', 'route']
        patients = patients[cols]
        return patients

    def parse_patient_info(self, string):
        patients = pd.DataFrame()

        date = re.search(r'\d*.\s*\d*\.\([가-힣]\)', string).group()

        region = re.search(r'([가-힣]|\s|\d|,)*확진자', string)
        if region is None:
            return None
        else:
            region = region.group()
        region = region.replace('확진자', '')

        region_split = region.split(' ')
        str_list = []
        num_list = []
        for split in region_split:
            if self.split_string_number(split) is not None:
                num_list.append(self.split_string_number(split)[0])
                if self.split_string_number(split)[1] != '' and \
                        self.split_string_number(split)[1] != '번' and \
                        self.split_string_number(split)[1] != '확진자':
                    str_list.append(self.split_string_number(split)[1])
            elif split != '' and split != '재양성':
                str_list.append(split)

        if not num_list:
            patient_id = re.search(r'확진자\s*\d*', string).group()
            patient_id = re.search(r'(\d*)$', patient_id).group()
            num_list.append(patient_id)

        if not num_list:
            return None

        for num in num_list:
            if num == '': continue
            patient_info = dict()
            patient_info.update({'confirmed_date': date})
            patient_info.update({'region': str_list[0]})
            patient_info.update({'patient_id': str_list[0] + num})
            patients = patients.append(patient_info, ignore_index=True)

        return patients

    def split_string_number(self, string):
        is_number_included = False
        is_string_included = False
        if re.findall(r'\d+', string):
            is_number_included = True
        if re.findall(r'[가-힣]+', string):
            is_string_included = True
        if is_number_included and is_string_included:
            number_part = re.findall(r'\d+', string)[0]
            string_part = re.findall(r'[가-힣]+', string)[0]
            if string_part == '번': string_part = ''
            return number_part, string_part

        return None

    def parse_patient_route(self, string):
        string = string.replace('<p class=\"iwa-exp\">', '')
        is_date_exist = False

        indices = [(m.start(0), m.end(0)) for m in re.finditer(r'<br\/>', string)]

        start = 0

        patients = pd.DataFrame()
        patient = dict()

        for index in indices:
            end = index[0]
            row = string[start:end].replace('<br/>', '')
            start = end

            if len(row) == 0: continue

            date = re.search(r'\[(\d|[가-힣]|\s|\(|\))*\]', row)
            if date is not None:
                patient.update({'date': date.group()[1:-1]})
                is_date_exist = True
                continue
            else:
                if is_date_exist is False: continue

            row = row.replace('\n', '')
            row = row.replace('\t', '')
            patient.update({'route': row})
            patients = patients.append(patient, ignore_index=True)

        return patients


class MapoParser1(Parser):
    def __init__(self, url, table_class):
        super(MapoParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        contents = soup.find('div', attrs={'class': self.table_class})
        div = contents.find('div', attrs={'class': 'tbl_wrap'})
        table = div.find('table')
        tbody = table.find('tbody')
        return tbody

    def extract_info(self, table):
        tr_list = table.find_all('tr')

        patients = pd.DataFrame()
        patient = dict()

        for tr in tr_list:
            td_list = tr.find_all('td')
            if len(td_list) == 0: continue

            patient.update({'patient_id': td_list[0].text.replace('\n', '')})
            patient.update({'confiemd_date': td_list[1].text.replace('\n', '')})
            patient_routes = str(td_list[2])
            patient_routes = self.parse_patient_route(patient_routes)
            for route in patient_routes:
                patient.update({'route': route})
                patients = patients.append(patient, ignore_index=True)

        return patients

    def parse_patient_route(self, string):
        pattern = '○(\d|\s|[가-힣]|\(|\)|\.|~|:)+'
        indices = [(m.start(0), m.end(0)) for m in re.finditer(pattern, string)]
        rows = []

        for index in indices:
            row = string[index[0]:index[1]]
            rows.append(row)

        return rows


class MapoParser2(Parser):
    def __init__(self, url, table_class):
        super(MapoParser2, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        contents = soup.find('div', attrs={'class': self.table_class})
        div = contents.find('div', attrs={'class': 'tbl_wrap'})
        table = div.find('table')
        tbody = table.find('tbody')
        return tbody

    def extract_info(self, table):
        tr_list = table.find_all('tr')

        patients = pd.DataFrame()
        patient = dict()

        for tr in tr_list:
            td_list = tr.find_all('td')
            if len(td_list) == 0: continue

            li_info = td_list[0].find_all('li')
            patient.update({'patient_id': li_info[0].text})
            patient.update({'confirmed_date': li_info[1].text})

            route_td = td_list[2]
            routes = self.parse_patient_route(str(route_td))

            for key in patient:
                routes[key] = patient[key]

            patients = patients.append(routes, ignore_index=True)

        cols = ['patient_id', 'confirmed_date', 'date', 'route']
        patients = patients[cols]

        return patients

    def parse_patient_route(self, string):
        p_tag = re.search(r'<p(.*)<\/p>', string)
        if p_tag is not None:
            string = string.replace(p_tag.group(), '')

        indices = [(m.start(0), m.end(0)) for m in re.finditer(r'<\/*[a-z]+\/*>', string)]
        start = indices[0][1]

        rows = []
        for index in indices[1:]:
            row = string[start:index[0]]
            start = index[1]

            if re.search(r'^\s*[가-힣]', row) is not None:
                rows[-1] += ' %s' % row
                continue

            if re.search(r'<p class="c_txt_b">', row) is not None:
                row = row[:re.search(r'<p class="c_txt_b">', row).span()[0]]

            if re.search(r'^\s*※', row) is not None:
                continue

            rows.append(row)

        patients = pd.DataFrame()

        for row in rows:
            patient = self.split_date_route(row)
            if patient is not None: patients = patients.append(patient, ignore_index=True)

        return patients

    def split_date_route(self, string):
        time_pattern = '\d*:\d*'
        day_pattern = '\d*.\s*\d*\([가-힣]\)'
        time_indices = [(m.start(0), m.end(0)) for m in re.finditer(time_pattern, string)]
        day_indices = [(m.start(0), m.end(0)) for m in re.finditer(day_pattern, string)]

        time_last = 0 if len(time_indices) == 0 else time_indices[-1][1]
        day_last = 0 if len(day_indices) == 0 else day_indices[-1][1]

        if time_last == 0 and day_last == 0: return None

        last = time_last if time_last > day_last else day_last

        time = string[:last].replace('\t', '')
        time = time.replace('\n', '')
        route = string[last:].replace('\t', '')
        route = route.replace('\n', '')

        if len(route) == 0: return None

        patient = dict()
        patient.update({'date': time})
        patient.update({'route': route})

        return patient


class SeodaemoonParser1(Parser):
    def __init__(self, url, table_class):
        super(SeodaemoonParser1, self).__init__(url, table_class)

    def read_html(self, path):
        with open(path, 'r', encoding='cp949') as f:
            text = f.read()

        return text

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        tables = soup.find_all('table', attrs={'class': self.table_class})
        tbodies = []
        for table in tables:
            tbodies.append(table.find('tbody'))
        return tbodies

    def extract_info(self, tables):
        patients = pd.DataFrame()

        for table in tables:
            tr_list = table.find_all('tr')
            patient = dict()

            for tr in tr_list:
                td_list = tr.find_all('td')
                if len(td_list) < 2: continue
                if '확진자에 대한 역학조사 진행 중입니다' in td_list[2].text:
                    continue

                if tr.find('p', attrs={'style': 'text-align:center'}) is not None:
                    patient.update({'patent_id': td_list[0].text})
                    td_list = td_list[2:]

                if len(td_list) < 4: continue

                patient.update({'type': td_list[0].text})
                patient.update({'route': td_list[1].text})
                patient.update({'location': td_list[2].text})
                patient.update({'date': td_list[3].text})

                if re.search(r'^\s*$', td_list[3].text) is not None:
                    continue

                patients = patients.append(patient, ignore_index=True)

        return patients


class SeochoParser1(Parser):
    def __init__(self, url, table_class):
        super(SeochoParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        div_list = soup.find_all('div', attrs={'class': self.table_class})

        tbodies = []
        for div in div_list:
            tbodies.append(div.find('tbody'))

        return tbodies

    def extract_info(self, tables):
        patients = pd.DataFrame()

        for i, table in enumerate(tables):
            tr_list = table.find_all('tr')
            patient = dict()

            for tr in tr_list:
                td_list = tr.find_all('td')

                if len(td_list) < 2: continue

                patient_info = td_list[0].text
                region = re.search(r'\([가-힣|\d]*\)', patient_info)
                if region is None:
                    patient_id = patient_info
                    region = '서초구'
                else:
                    region = region.group()
                    patient_id = patient_info.replace(region, '')
                    region = region[1:-1]

                routes = td_list[1].text
                routes = routes.split('\n')
                reason = routes[0]

                if re.search(r'^\s*-', reason) is not None:
                    reason = '감염경로 모름'
                elif re.search(r'^\s*[ｏ|\d]', reason) is not None:
                    reason = '감염경로 모름'
                else:
                    routes = routes[1:]

                patient.update({'patient_id': patient_id})
                patient.update({'region': region})
                patient.update({'reason': reason})

                date = ''
                for j, route in enumerate(routes):
                    if re.search(r'^\*', route) is not None: continue
                    if re.search(r'^⇒', route) is not None: continue
                    if re.search(r'^\s*-', route):
                        date = route
                        if re.search(r'([가-힣]|\s)*$', date) is not None:
                            route = re.search(r'([가-힣]|\s|\.)*$', date).group()
                            if len(route) == 0: continue
                            date = date.replace(route, '')
                            patient.update({'date': date})
                            patient.update({'route': route})
                            patients = patients.append(patient, ignore_index=True)
                            continue
                        continue
                    patient.update({'date': date})
                    patient.update({'route': route})
                    patients = patients.append(patient, ignore_index=True)

        return patients


class SeongdongParser1(Parser):
    def __init__(self, url, table_class):
        super(SeongdongParser1, self).__init__(url, table_class)

    def read_html(self, path):
        with open(path, 'r', encoding='cp949') as f:
            text = f.read()

        return text

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        div_list = []
        div_list.append(soup.find('div', attrs={'class': self.table_class}))
        table_class = self.table_class + ' box_color1'
        div_list.append(soup.find('div', attrs={'class': table_class}))

        return div_list

    def extract_info(self, div_list):
        patients = pd.DataFrame()

        for i, outer_div in enumerate(div_list):
            div = outer_div.find('div', attrs={'class': 'itembox'})
            li_list = div.find_all('li', attrs={'class': 'item'})

            for li in li_list:
                patient = dict()

                reason = re.search(r'(?<=감염경로\s:\s)([가-힣]|\s|\d|#|,|\(|\))*', str(li))
                if reason is None:
                    reason = '감염경로 모름'
                else:
                    reason = reason.group()
                    reason = reason.replace('\n', '')

                patient.update({'reason': reason})

                body = li.find('tbody')
                if body is None: continue

                tr_list = body.find_all('tr')
                for j, tr in enumerate(tr_list):
                    td_list = tr.find_all('td')
                    if j == 0:
                        patient_info = td_list[0]
                        pattern = r'(?<=<strong>)([가-힣]|\s|\d|\(|\))*(?=<)'
                        patient_info = re.search(pattern, str(patient_info)).group()
                        patient.update({'patient_id': patient_info})
                        td_list = td_list[1:]

                    if len(td_list) == 1: continue

                    date = td_list[0].text
                    patient.update({'date': date})

                    pattern = r'(?<=<\/b>)([가-힣]|\s|\d|\(|\)|-|[a-z]|[A-Z]|&|;)*'
                    routes = [x.group() for x in re.finditer(pattern, str(td_list[1]))]
                    for route in routes:
                        patient.update({'route': route})
                        patients = patients.append(patient, ignore_index=True)

        return patients


class SeongdongParser2(SeongdongParser1):
    def __init__(self, url, table_class):
        super(SeongdongParser2, self).__init__(url, table_class)

    def extract_info(self, div_list):
        patients = pd.DataFrame()

        for outer_div in div_list:
            li_list = outer_div.find_all('li', attrs={'class': 'item'})

            for li in li_list:
                patient = dict()
                div_title = li.find('div', attrs={'class': 'titlebox'})
                div_layer = li.find('div', attrs={'class': 'layer'})

                patient_id = div_title.find('p').text
                patient.update({'patient_id': patient_id})

                body = div_layer.find('tbody')
                if body is None: continue
                tr_list = body.find_all('tr')[1:]
                for tr in tr_list:
                    td_list = tr.find_all('td')
                    if len(td_list) < 6: break

                    location_type = td_list[1].text
                    location = td_list[2].text
                    address = td_list[3].text
                    date = td_list[4].text
                    patient.update({'location_type': location_type})
                    patient.update({'location': location})
                    patient.update({'address': address})
                    patient.update({'date': date})
                    patients = patients.append(patient, ignore_index=True)

        cols = ['patient_id', 'date', 'location', 'location_type', 'address']
        patients = patients[cols]
        return patients


class SeongbukParser1(Parser):
    def __init__(self, url, table_class):
        super(SeongbukParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('tbody')

        return table

    def extract_info(self, table):
        patients = pd.DataFrame()

        tr_list = table.find_all('tr')
        for i in range(0, len(tr_list), 2):
            patient = dict()
            patient_info = tr_list[i]
            info_td_list = patient_info.find_all('td')
            patient_id = info_td_list[0].text
            patient_id = patient_id.replace('\n', '')
            reason = info_td_list[2].text
            region = info_td_list[4].text
            patient.update({'patient_id': patient_id})
            patient.update({'reason': reason})
            patient.update({'region': region})

            pattern = r'(?<=>)(\s|-|\d|[a-z]|[A-Z]|[가-힣]|\(|\)|~|:|,|○)*(?=<br)'
            routes = [x.group() for x in re.finditer(pattern, str(tr_list[i + 1]))]
            date = ''
            for route in routes:
                route = route.replace('\n', '')
                route = re.sub(r'\s{2,}', ' ', route)
                if re.search(r'^-', route) is not None:
                    date = route
                    patient.update({'date': date})
                    continue
                patient.update({'route': route})
                patients = patients.append(patient, ignore_index=True)

        return patients


class SongpaParser1(Parser):
    def __init__(self, url, table_class):
        super(SongpaParser1, self).__init__(url, table_class)

    def read_html(self, path):
        path = path.replace('.html', '')
        path += '_files\\table.html'

        with open(path, 'r', encoding='cp949') as f:
            text = f.read()

        return text

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('tbody')

        return table

    def extract_info(self, table):
        patients = pd.DataFrame()

        tr_list = table.find_all('tr')
        for tr in tr_list:
            patient = dict()
            td_list = tr.find_all('td')
            if len(td_list) < 7: continue

            patient_id = td_list[0].text
            address = td_list[2].text
            place_type = td_list[3].text
            patient.update({'patient_id': patient_id})
            patient.update({'address': address})
            patient.update({'place_type': place_type})

            date = None

            if len(td_list) == 7:
                date = td_list[5]
            elif len(td_list) == 8:
                location = td_list[5].text
                patient.update({'exact_address': location})
                date = td_list[6]

            if date is None: continue
            pattern = r'(?<=>)([~\d|\.|\(|[가-힣]|\)|\s|:|])*(?=<)'
            dates = [x.group() for x in re.finditer(pattern, str(date))]
            for new_date in dates:
                patient.update({'date': new_date})
                patients = patients.append(patient, ignore_index=True)

        return patients


class YeongdeungpoParser1(Parser):
    def __init__(self, url, table_class):
        super(YeongdeungpoParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        section = soup.find('section', self.table_class)
        tables = section.find_all('table')
        return tables

    def extract_info(self, tables):
        patients = pd.DataFrame()

        for table in tables:
            tbody = table.find('tbody')
            if tbody is None: continue

            notice_list1 = tbody.find_all('p')
            notice_list2 = tbody.find_all('td')
            if notice_list1 is None: continue
            if notice_list2 is not None:
                notice_list1 += notice_list2

            for notice in notice_list1:
                notice = notice.text
                if '2) 이동경로' not in notice: continue

                info_pattern = r'(?<=1\) 현황)[^"]*?(?=2\) 이동경로)'
                patient_info_list = [x.group() for x in re.finditer(info_pattern, notice)]

                route_pattern = r'(?<=2\) 이동경로)[^"]*?(?=3\) 조치사항)'
                route_list = [x.group() for x in re.finditer(route_pattern, notice)]

                patient = dict()
                for i, patient_info in enumerate(patient_info_list):
                    patient_info = re.search(r'[^"]*?(?=확진자)', patient_info)
                    patient_id = patient_info.group()
                    patient_id = patient_id.replace('\n', '')
                    patient.update({'patient_id': patient_id})

                    route = route_list[i]
                    route_splitted = route.split('\n')
                    for route_elem in route_splitted:
                        if re.search(r'^\s*$', route_elem) is not None: continue
                        if '중입니다' in route_elem: continue

                        is_date = re.search(r'^\s*•', route_elem)
                        if is_date is not None:
                            patient.update({'date': route_elem})
                        else:
                            patient.update({'route': route_elem})
                            patients = patients.append(patient, ignore_index=True)

        return patients


class YongsanParser1(Parser):
    def __init__(self, url, table_class):
        super(YongsanParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')

        tables = []
        div_list = soup.find_all('div', attrs={'class': self.table_class})
        for div in div_list:
            table = div.find('table')
            tables.append(table)

        return tables

    def extract_info(self, tables):
        patients = pd.DataFrame()

        for table in tables:
            body = table.find('tbody')
            info_tr_list = body.find_all('tr', attrs={'class': 'corona_info'})
            if len(info_tr_list) == 0:
                info_tr_list = body.find_all('tr', attrs={'class': 'corona_info1'})
            detail_tr_list = body.find_all('tr', attrs={'class': 'coronaV-detail-cell'})

            for i in range(len(info_tr_list)):
                patient = dict()
                info_tr = info_tr_list[i]
                detail_tr = detail_tr_list[i]
                route_table = detail_tr.find('table', attrs={'class': 'sub_table'})
                if route_table is None: continue
                route_body = route_table.find('tbody')
                route_tr_list = route_body.find_all('tr')

                td_list = info_tr.find_all('td')
                patient.update({'patient_id': td_list[0].text})
                patient.update({'region': td_list[2].text})

                for route_tr in route_tr_list:
                    td_list = route_tr.find_all('td')
                    location_region = td_list[0].text
                    location_type = td_list[1].text
                    location_address = td_list[2].text
                    date = td_list[3].text
                    date = date.replace('\n', '')
                    date = date.replace('\t', '')

                    patient.update({'location_region': location_region})
                    patient.update({'location_type': location_type})
                    patient.update({'location_address': location_address})
                    patient.update({'date': date})

                    patients = patients.append(patient, ignore_index=True)

        return patients


class EunpyeongParser1(Parser):
    def __init__(self, url, table_class):
        super(EunpyeongParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        div = soup.find('div', attrs={'id': self.table_class})
        table = div.find('table')
        body = table.find('tbody')
        return body

    def extract_info(self, table):
        patients = pd.DataFrame()

        tr_list = table.find_all('tr')

        index = 0
        while True:
            if index + 1 >= len(tr_list): break

            patient = dict()
            info_tr = tr_list[index]
            route_tr = tr_list[index + 1]

            if info_tr.get('class') == 'swtr': break
            index += 2

            inner_table = route_tr.find('table')
            if inner_table is None: continue
            inner_body = inner_table.find('tbody')
            inner_tr_list = inner_body.find_all('tr')

            th = info_tr.find('th').text
            th = th.replace('\n', '')
            th = re.sub(r'\s{2,}', ' ', th)
            patient.update({'patient_id': th})

            for inner_tr in inner_tr_list:
                td_list = inner_tr.find_all('td')
                location_region = td_list[0].text
                location_type = td_list[1].text
                location_address = td_list[2].text
                date = td_list[3].text

                patient.update({'location_region': location_region})
                patient.update({'location_type': location_type})
                patient.update({'location_address': location_address})
                patient.update({'date': date})

                patients = patients.append(patient, ignore_index=True)

        return patients


class JongnoParser1(Parser):
    def __init__(self, url, table_class):
        super(JongnoParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        div = soup.find('div', attrs={'class': self.table_class})
        table = div.find('table')
        body = table.find('tbody')
        return body

    def extract_info(self, table):
        patients = pd.DataFrame()

        tr_list = table.find_all('tr')

        index = 0
        patient = dict()
        while True:
            if index >= len(tr_list): break

            tr = tr_list[index]
            index += 1

            td_list = tr.find_all('td')
            if td_list[0].get('rowspan') == '3' or td_list[0].get('rowspan') == '4':
                patient_id = td_list[0].text
                reason = re.search(r'.*(?=\/)', td_list[1].text)
                reason = td_list[1].text if reason is None else reason.group()
                patient.update({'patient_id': patient_id})
                patient.update({'reason': reason})
            else:
                if len(td_list) != 5: continue
                location_type = td_list[0].text
                location_type = location_type.replace('\n', '')
                location_type = re.sub(r'\s{2,}', '', location_type)
                location_name = td_list[1].text
                location_address = td_list[2].text
                date = td_list[3].text
                date = date.replace('\n', '')
                date = re.sub(r'\s{2,}', '', date)

                patient.update({'location_type': location_type})
                patient.update({'location_name': location_name})
                patient.update({'location_address': location_address})
                patient.update({'date': date})

                patients = patients.append(patient, ignore_index=True)

        cols = ['patient_id', 'date', 'reason', 'location_type', 'location_name', 'location_address']
        patients = patients[cols]
        return patients


class JungParser1(Parser):
    def __init__(self, url, table_class):
        super(JungParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        div = soup.find('div', attrs={'class': self.table_class})
        return div

    def extract_info(self, div):
        patients = pd.DataFrame()

        dt_list = div.find_all('dt')
        dd_list = div.find_all('dd')

        for i, dd in enumerate(dd_list):
            table = dd.find('table')
            if table is None: continue

            patient = dict()

            patient_id = dt_list[i].text
            patient_id = patient_id.replace('\n', '')
            patient_id = re.sub(r'\s{2,}', '', patient_id)
            patient.update({'patient_id': patient_id})

            tbody = table.find('tbody')
            tr_list = tbody.find_all('tr')
            for tr in tr_list:
                td_list = tr.find_all('td')

                if len(td_list) < 4: continue

                location_region = td_list[0].text
                location_region = location_region.replace('\n', '')
                location_region = re.sub(r'\s{2,}', '', location_region)
                location_type = td_list[1].text
                location_address = td_list[2].text
                location_address = location_address.replace('\n', '')
                location_address = re.sub(r'\s{2,}', '', location_address)
                date = td_list[3].text
                date = date.replace('\n', '')
                date = re.sub(r'\s{2,}', '', date)

                patient.update({'location_region': location_region})
                patient.update({'location_type': location_type})
                patient.update({'location_address': location_address})
                patient.update({'date': date})

                patients = patients.append(patient, ignore_index=True)

        if len(patients) == 0: return patients

        cols = ['patient_id', 'date', 'location_type', 'location_region', 'location_address']
        patients = patients[cols]
        return patients


class JungnangParser1(Parser):
    def __init__(self, url, table_class):
        super(JungnangParser1, self).__init__(url, table_class)

    def extract_table(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        div = soup.find('div', attrs={'id': self.table_class})
        return div

    def extract_info(self, div):
        patients = pd.DataFrame()
        patient = dict()

        p_list = div.find_all('p')
        patient_id = ''
        for p in p_list:
            p_regexed1 = re.search(r'.*(?=확진자)', p.text)
            p_regexed2 = re.search(r'#', p.text)
            if p_regexed1 is not None and p_regexed2 is not None:
                patient_id = p_regexed1.group()
                break
        if len(patient_id) == 0: patient_id = 'unknown'
        patient.update({'patient_id': patient_id})

        table = div.find('table')
        body = table.find('tbody')

        tr_list = body.find_all('tr')[1:]
        for tr in tr_list:
            td_list = tr.find_all('td')

            location_region = td_list[0].text
            location_type = td_list[1].text
            location_address = td_list[2].text
            date = td_list[3].text

            patient.update({'location_region': location_region})
            patient.update({'location_type': location_type})
            patient.update({'location_address': location_address})
            patient.update({'date': date})

            patients = patients.append(patient, ignore_index=True)

        return patients


class JungnangParser2(JungnangParser1):
    def __init__(self, url, table_class):
        super(JungnangParser2, self).__init__(url, table_class)

    def remove_spaces(self, element):
        element = element.replace('\n', '')
        element = re.sub(r'\s{2,}', ' ', element)
        return element

    def extract_info(self, div):
        patients = pd.DataFrame()

        prev_tag = None
        for tag in div():
            if tag.name == 'p' and '확진자' in tag.text:
                prev_tag = tag
            if prev_tag is None or tag.name != 'table':
                continue

            if '확진자' in prev_tag.text:
                patient = dict()
                patient_id = re.search(r'.*(?=확진자)', prev_tag.text).group()
                patient.update({'patient_id': patient_id})

                table = tag.find('tbody')
                tr_list = table.find_all('tr')

                column_len = len(tr_list[0].find_all('td'))
                column_names = ['location_type', 'location_address', 'date', 'is_infected']
                if column_len == 5: column_names.insert(0, 'location_region')

                rowspan_list = [0 for i in range(column_len)]
                tr_list = tr_list[1:]

                for j, tr in enumerate(tr_list):
                    iter_list = []
                    td_list = tr.find_all('td')

                    for i in range(column_len):
                        if rowspan_list[i] > 0:
                            rowspan_list[i] -= 1
                        if rowspan_list[i] == 0:
                            iter_list.append(i)

                    for i in range(len(iter_list)):
                        if td_list[i].has_attr('rowspan'):
                            rowspan_list[iter_list[i]] = int(td_list[i]['rowspan'])

                        element = td_list[i].text
                        element = self.remove_spaces(element)
                        patient.update({column_names[iter_list[i]]: element})

                    patients = patients.append(patient, ignore_index=True)

        return patients


if __name__ == '__main__':
    url = 'https://www.seoul.go.kr/coronaV/coronaStatus.do'
    table_class = 'tstyle05 tstyleP status-datatable datatable-multi-row'

    parser = StatusParser(url, table_class)
    root = Path(os.getcwd()).parent

    content_path = os.path.join(root, 'data', 'corona_status.html')
    content = parser.read_html(content_path)

    table = parser.extract_table(content)

    info_path = os.path.join(root, 'data', 'corona_status.csv')
    patient_info = parser.extract_info(table)
    parser.save_info(info_path, patient_info)
