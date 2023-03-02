from pyhdf.SD import SD, SDC
import math
import pandas as pd

def convert_to_float(number):
    return float(number)
#poleznaya funkziya

filename_03 = '20220404_074331_TERRA_MOD03.hdf'
filename_14 = '20220404_074331_TERRA_MOD14.hdf'
#imena

hdf_03 = SD(filename_03, SDC.READ)
hdf_14 = SD(filename_14, SDC.READ)
#otkrivayut fayli, format pyhdf



FP_row = hdf_14.select('FP_line')
FP_col = hdf_14.select('FP_sample')
#poluchayut stroki po imeni

for i in range(len(FP_row)):
    FP_col[i] = int(FP_col[i])
    FP_row[i] = int(FP_row[i])
#perevod v int. pochemu bez funkzii, kak dlya floata?

Longitude = hdf_03.select('Longitude')
Latitude = hdf_03.select('Latitude')
#poluchayut stroki po imeni

with open('Krasnodar_Krai.txt', 'r') as reader:
    text = reader.readlines()
#otkrivayut zaraniye podgotovlenniy fayl. tipa vector

colobok = []
#/t-tab
columns = text[0].split('\t')
for line in text[1::]:
    colobok.append(list(map(float, line.split('\t'))))
#obrezanniy snimok, hz, kakoy kanal

bands_dict = {
    'Pixel-X': [],
    'Pixel-Y': [],
    'Center_Latitude': [],
    'Center_Longitude': [],
    'EV_1KM_Emissive_21': [],
    'EV_1KM_Emissive_22': [],
    'EV_1KM_Emissive_31': [],
    'EV_1KM_Emissive_32': [],
    'EV_250_Aggr1km_RefSB_1': []
}
#sozdaet pustoy slovar'

"""
Pixel-X
Pixel-Y
Center_Latitude
Center_Longitude
EV_1KM_Emissive_21
EV_1KM_Emissive_22
EV_1KM_Emissive_31
EV_1KM_Emissive_32
EV_250_Aggr1km_RefSB_1
"""

for row in colobok:
    for idx, value in enumerate(row):
        if columns[idx] == 'Pixel-X':
            bands_dict['Pixel-X'].append(value)
        elif columns[idx] == 'Pixel-Y':
            bands_dict['Pixel-Y'].append(value)
        elif columns[idx] == 'Latitude':
            bands_dict['Center_Latitude'].append(value)
        elif columns[idx] == 'Longitude':
            bands_dict['Center_Longitude'].append(value)
        elif columns[idx] == 'EV_1KM_Emissive_21':
            bands_dict['EV_1KM_Emissive_21'].append(value)
        elif columns[idx] == 'EV_1KM_Emissive_22':
            bands_dict['EV_1KM_Emissive_22'].append(value)
        elif columns[idx] == 'EV_1KM_Emissive_31':
            bands_dict['EV_1KM_Emissive_31'].append(value)
        elif columns[idx] == 'EV_1KM_Emissive_32':
            bands_dict['EV_1KM_Emissive_32'].append(value)
        elif columns[idx] == 'EV_250_Aggr1km_RefSB_1':
            bands_dict['EV_250_Aggr1km_RefSB_1'].append(value)
        else:
            pass
#perepisivaet soderzhanie massiva v slovar'
data = {
    'IMAGEID': [],
    'N': [],
    'col': [],
    'row': [],
    'longitude': [],
    'latitude': [],
    'T4': [],
    'T11': [],
    'pixel_poly': [],
    'region': []
}
#eshe odin slovar'

k = 1
mod14_check = 0
for i in range(len(bands_dict['Pixel-Y'])):
    col = int(bands_dict['Pixel-X'][i] - 0.5)
    row = int(bands_dict['Pixel-Y'][i] - 0.5)
    if bands_dict['EV_1KM_Emissive_22'][i] < 0 or bands_dict['EV_1KM_Emissive_31'][i] < 0 or bands_dict['EV_1KM_Emissive_21'][i] < 0:
        continue
    t4 = (1.4387752 * 10 ** 4 / 3.959) / (math.log(1 + (1.19104282 * 10 ** 8 * 3.959 ** (-5)) / bands_dict['EV_1KM_Emissive_21'][i]))
    t11 = (1.4387752 * 10 ** 4 / 11.03) / (math.log(1 + (1.19104282 * 10 ** 8 * 11.03 ** (-5)) / bands_dict['EV_1KM_Emissive_31'][i]))
    dt = abs(t4-t11)
    if bands_dict['EV_1KM_Emissive_22'][i] >= 0.74 and 6.9 <= bands_dict['EV_1KM_Emissive_32'][i] <= 10 and dt >= 12 and \
            bands_dict['EV_250_Aggr1km_RefSB_1'][i] < 0.112:

        lat_c = bands_dict['Center_Latitude'][i]
        lon_c = bands_dict['Center_Longitude'][i]

        N = k
        k += 1
        n = row
        m = col



        latitude = Latitude[row][col]
        longitude = Longitude[row][col]

        lat_r = Latitude[row][col+1]
        lon_r = Longitude[row][col + 1]
        lat_d = Latitude[row + 1][col]
        lon_d = Longitude[row + 1][col]
        lat_diag = Latitude[row + 1][col + 1]
        lon_diag = Longitude[row + 1][col + 1]

        lat1 = float(latitude)
        lon1 = float(longitude)
        lat2 = float(lat_r)
        lon2 = float(lon_r)
        lat3 = float(lat_diag)
        lon3 = float(lon_diag)
        lat4 = float(lat_d)
        lon4 = float(lon_d)

        if not lat3 <= lat_c <= lat1:
            print('INVALID GEOMETRY')

        wkt = f'POLYGON (({lon1} {lat1}, {lon2} {lat2}, {lon3} {lat3}, {lon4} {lat4}, {lon1} {lat1}))'

        data['IMAGEID'].append('20220404_074331_TERRA')
        data['N'].append(N)
        data['col'].append(m)
        data['row'].append(n)
        data['longitude'].append(lon_c)
        data['latitude'].append(lat_c)
        data['T4'].append(round(t4, 1))
        data['T11'].append(round(t11, 1))
        data['pixel_poly'].append(wkt)
        data['region'].append('region')

        print(N, m, n, longitude, latitude, round(t4, 1), round(t11, 1) )

        if col in FP_col and row in FP_row:
            mod14_check += 1
print()
print(f'Number of pixels checked by mod14: {mod14_check}')
df = pd.DataFrame(data=data, columns=data.keys())
# print(df)
df.to_csv('20220319_074230_TERRA_CubeSats.csv', sep=',', index=False)