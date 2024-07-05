import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Load the flight record file
file_path = 'DJIFlightRecord_2024-06-20_[09-22-37].csv'

# Load the CSV file with low_memory=False to avoid DtypeWarning
# The skiprows code is due to the first line of this version of csv files exported is irrelevant to the data
df = pd.read_csv(file_path, sep=',', skiprows=1,low_memory=False)

# Extract the header
header = df.columns

# Print the header in a formal format
print("CSV File Header:")
print("================")
for column in header:
    print(f"- {column}")

# Extract the relevant columns
timeInSecond = df["OSD.flyTime [s]"]
DroneLatitude = df["OSD.latitude"]
DroneLongitude = df["OSD.longitude"]

# Plot the data
plt.plot(DroneLatitude, DroneLongitude, label="Drone Coordinates")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Flying Locus of drone")

plt.legend()
plt.show()



# WGS-84 ellipsiod parameters
a = 6378137.0  # semi-major axis in meters
f = 1 / 298.257223563  # flattening
b = (1 - f) * a  # semi-minor axis

def vincenty_distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    L = lon2 - lon1
    U1 = math.atan((1 - f) * math.tan(lat1))
    U2 = math.atan((1 - f) * math.tan(lat2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)
    
    lamb = L
    for _ in range(1000):  # limit the number of iterations to avoid infinite loop
        sin_lambda = math.sin(lamb)
        cos_lambda = math.cos(lamb)
        sin_sigma = math.sqrt((cosU2 * sin_lambda) ** 2 + 
                              (cosU1 * sinU2 - sinU1 * cosU2 * cos_lambda) ** 2)
        if sin_sigma == 0:
            return 0  # coincident points
        
        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cosU1 * cosU2 * sin_lambda / sin_sigma
        cos2_alpha = 1 - sin_alpha ** 2
        cos2_sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos2_alpha
        
        if math.isnan(cos2_sigma_m):
            cos2_sigma_m = 0  # equatorial line

        C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
        lamb_prev = lamb
        lamb = L + (1 - C) * f * sin_alpha * (sigma + C * sin_sigma *
                                               (cos2_sigma_m + C * cos_sigma *
                                                (-1 + 2 * cos2_sigma_m ** 2)))
        
        if abs(lamb - lamb_prev) < 1e-12:
            break
    else:
        return None  # formula failed to converge

    u2 = cos2_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    delta_sigma = B * sin_sigma * (cos2_sigma_m + B / 4 *
                                   (cos_sigma * (-1 + 2 * cos2_sigma_m ** 2) -
                                    B / 6 * cos2_sigma_m *
                                    (-3 + 4 * sin_sigma ** 2) *
                                    (-3 + 4 * cos2_sigma_m ** 2)))
    
    s = b * A * (sigma - delta_sigma)
    
    return s



# Calculate the distance in meters between two points using the Haversine formula
def calc_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000.0
    
    # Convert latitude and longitude from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in meters
    distance = R * c
    return distance

# Calculate the distance between the two points
# 7.65m*（8.5+4.5）m

length = vincenty_distance(3.0632653, 101.6007073, 3.0635232, 101.6007094)
width = vincenty_distance(3.0635232, 101.6007094, 3.0635269,101.6009070)
print(f"Length of building: {length} meters")
print(f"Width of building: {width} meters")