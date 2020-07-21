import pandas as pd
import numpy as np
from area import area
import statsmodels.api as sm

# change path to the dataset
MT_dataset_path = 'MT_cleaned.csv'
VT_dataset_path = 'VT_cleaned.csv'
df_MT = pd.read_csv(MT_dataset_path, dtype="string")
df_VT = pd.read_csv(VT_dataset_path, dtype="string")
rows_count_MT = len(df_MT.index)
rows_count_VT = len(df_VT.index)
column_names_MT = list(df_MT.columns)

# Q2-1
num_males = (df_MT.driver_gender == "M").sum()
proportion_male_drivers = num_males / rows_count_MT
print('Question 2-1:')
print('Proportion of male drivers in MT:', proportion_male_drivers)
print('---------------------------------------------------------')

# Q2-2
num_arrested = (df_MT.is_arrested == "TRUE").sum()
arresting_likelihood = num_males / rows_count_MT


# Q2-3
num_speed_violation = (df_MT['violation'].str.contains("Speeding")).sum()
proportion_speed_violation = num_speed_violation / rows_count_MT
print('Question 2-3:')
print('Proportion of speed violation in MT:', proportion_speed_violation)
print('---------------------------------------------------------')

# Q2-4
DUI_violation_MT = (df_MT['violation'].str.contains("DUI")).sum()
DUI_violation_likelihood_MT = DUI_violation_MT / rows_count_MT
DUI_violation_VT = (df_VT['violation'].str.contains("DUI")).sum()
DUI_violation_likelihood_VT = DUI_violation_VT / rows_count_VT

factor_increase_MT_over_VT = DUI_violation_likelihood_MT / DUI_violation_likelihood_VT
print('Question 2-4:')
print('Factor increase in traffic stop DUI likelihood in MT over VT:', factor_increase_MT_over_VT)
print('---------------------------------------------------------')

# Q2-5
df_MT_stopdate_vehicleyear = df_MT[['stop_date', 'vehicle_year']].dropna()

stop_year = df_MT_stopdate_vehicleyear['stop_date'].str[0:4].astype(float).to_numpy()
vehicle_year_numeric = pd.to_numeric(df_MT_stopdate_vehicleyear['vehicle_year'], errors='coerce')
nan_vehicle_year_numeric = vehicle_year_numeric.isna()
stop_year = stop_year[~nan_vehicle_year_numeric]
manufacture_year = vehicle_year_numeric[~nan_vehicle_year_numeric].to_numpy()
X = stop_year
y = manufacture_year
X2 = sm.add_constant(X)
estimator = sm.OLS(y, X2)
estimator_fitted = estimator.fit()
pvalue = estimator_fitted.f_pvalue
ypred = estimator_fitted.predict(np.array([[1., 2020.]]))
print('Question 2-5:')
print('Average manufacture year of vehicles stopped in MT in 2020:', ypred[0])
print('P-value of linear regression:', pvalue)
print('---------------------------------------------------------')

# Q2-6
stop_hour = df_MT['stop_time'].dropna().str[0:2].astype(int).to_numpy()
bin_edges = np.arange(0., 23. + 2.) - 0.5
hist, _ = np.histogram(stop_hour, bins=bin_edges)
# max_hour_arg = np.argmax(hist)
# min_hour_arg = np.argmin(hist)
max_hour = np.max(hist)
min_hour = np.min(hist)
difference_MT = max_hour - min_hour

stop_hour = df_VT['stop_time'].dropna().str[0:2].astype(int).to_numpy()
bin_edges = np.arange(0., 23. + 2.) - 0.5
hist, _ = np.histogram(stop_hour, bins=bin_edges)
# max_hour_arg = np.argmax(hist)
# min_hour_arg = np.argmin(hist)
max_hour = np.max(hist)
min_hour = np.min(hist)
difference_VT = max_hour - min_hour

print('Question 2-6:')
print('Difference of total number of stops between max and min hours in MT:', difference_MT)
print('Difference of total number of stops between max and min hours in VT:', difference_VT)
print('---------------------------------------------------------')

# Q2-7


def clean_outliers(lat_lon_numpy, std_factor):
    lat_lon_mean = np.mean(lat_lon_numpy, axis=0)
    lat_lon_std = np.std(lat_lon_numpy, axis=0)
    lat_lon_lower_lim = lat_lon_mean - std_factor * lat_lon_std
    lat_lon_higher_lim = lat_lon_mean + std_factor * lat_lon_std
    outlier_condition = (lat_lon_numpy[:, 0] < lat_lon_lower_lim[0]) | (lat_lon_numpy[:, 0] > lat_lon_higher_lim[0]) | \
                        (lat_lon_numpy[:, 1] < lat_lon_lower_lim[1]) | (lat_lon_numpy[:, 1] > lat_lon_higher_lim[1])
    lat_lon_numpy_cleaned = lat_lon_numpy[~outlier_condition]
    # eliminated_records = lat_lon_numpy[outlier_condition]
    return lat_lon_numpy_cleaned


df_MT_county_lat_lon = df_MT[['county_name', 'lat', 'lon']].dropna()
county_name_unique_MT = df_MT_county_lat_lon['county_name'].unique()
areas = np.zeros((len(county_name_unique_MT)))
for i_county, county in enumerate(county_name_unique_MT):
    df_temp = df_MT_county_lat_lon[df_MT_county_lat_lon.county_name == county]
    lat_lon_numpy = df_temp[['lat', 'lon']].astype(float).values
    lat_lon_numpy_cleaned1 = clean_outliers(lat_lon_numpy, std_factor=5)
    lat_lon_numpy_cleaned2 = clean_outliers(lat_lon_numpy_cleaned1, std_factor=8)
    lat_lon = lat_lon_numpy_cleaned2.tolist()
    obj = {'type': 'Polygon', 'coordinates': [lat_lon]}
    areas[i_county] = area(obj) / 1000000.0

largest_county_ind = np.argmax(areas)
largest_county_area = areas[largest_county_ind]
largest_county = county_name_unique_MT[largest_county_ind]
print('Question 2-7:')
print(largest_county, 'is the largest county in MT')
print('with Area: ', largest_county_area, 'sq. km')
print('---------------------------------------------------------')


print('End of Code')
