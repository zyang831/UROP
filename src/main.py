import sys
import os
import tempfile

from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QStyleFactory
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pandas as pd
import folium
import numpy as np

# Constants
EARTH_RADIUS = 6371000  # radius of the Earth in meters

# Stylesheet
stylesheet = """
    QWidget {
        background-color: #333333;
        color: #FFFFFF;
    }

    QPushButton {
        background-color: #444444;
        border: none;
        color: #CCCCCC;
        padding: 8px 16px;
    }

    QPushButton:hover {
        background-color: #555555;
        color: #FFFFFF;
    }

    QLineEdit {
        background-color: #444444;
        border: none;
        color: #CCCCCC;
        padding: 4px 8px;
    }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        self.latitudes = []
        self.longitudes = []
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.selected_points = [] 
        self.initialize_ui()
        self.setStyleSheet(stylesheet)

    def initialize_ui(self):
            """Initializes the UI components.

            This method sets up the user interface components for the main window.
            It creates and configures the buttons, labels, and layout.
            The buttons are connected to their respective functions.
            The labels are initially set to display default messages.

            Args:
                None

            Returns:
                None
            """
            self.load_file_button = QPushButton("Load Flight Record")
            self.load_file_button.clicked.connect(self.load_file)
            self.plot_button = QPushButton("Plot Flight Path")
            self.plot_button.clicked.connect(self.plot_coordinates)
            self.measure_button = QPushButton("Measure")
            self.measure_button.clicked.connect(self.measure_selected_points)
            self.reset_button = QPushButton("Reset Plot")  
            self.reset_button.clicked.connect(self.reset_plot) 
            self.coordinates_label = QLabel("No coordinates loaded.")
            self.measurement_label = QLabel("No measurements taken.")

            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            layout.addWidget(self.load_file_button)
            layout.addWidget(self.plot_button)
            layout.addWidget(self.measure_button)
            layout.addWidget(self.reset_button) 
            layout.addWidget(self.coordinates_label)
            layout.addWidget(self.measurement_label)

            container = QWidget()
            container.setLayout(layout)
            self.setCentralWidget(container)

    def on_click(self, event):
            """Handles click events on the canvas, snapping to close existing points.

            Args:
                event (MouseEvent): The mouse event object containing information about the click event.

            Returns:
                None
            """
            CLOSENESS_THRESHOLD = 0.00001
            if event.inaxes:
                x, y = event.xdata, event.ydata
                # Snap to the closest existing point if within threshold
                x, y = self.get_closest_point_if_within_threshold(x, y, CLOSENESS_THRESHOLD)
                self.selected_points.append((x, y))
                self.coordinates_label.setText(f"Point selected at: Latitude={y}, Longitude={x}")
                self.plot_selected_points()

    def get_closest_point_if_within_threshold(self, x, y, threshold):
            """
            Finds the closest point to (x, y) within a given threshold.

            Parameters:
            - x (float): The x-coordinate of the point.
            - y (float): The y-coordinate of the point.
            - threshold (float): The maximum distance allowed for a point to be considered close.

            Returns:
            - closest_point (tuple): The closest point to (x, y) within the threshold, if found.
            - (x, y) (tuple): The original (x, y) coordinates if no point is found within the threshold.
            """
            closest_point = (x, y)
            min_distance = float('inf')
            for point in self.selected_points:
                distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
            if min_distance <= threshold:
                return closest_point
            else:
                return x, y

    def plot_selected_points(self):
            """Plots the selected points on the flight path.

            This method takes the selected points and plots them on the flight path.
            It retrieves the longitudes and latitudes from the selected points and
            plots them as red dots connected by a solid line.

            Returns:
                None
            """
            ax = self.figure.gca()
            longitudes, latitudes = zip(*self.selected_points) if self.selected_points else ([], [])
            ax.plot(longitudes, latitudes, 'ro-')
            self.canvas.draw()

    def measure_selected_points(self):
            """Measures the distances and area of the selected points.

            This method calculates the distances between the selected points and
            optionally calculates the area if there are more than two points selected.
            The distances are displayed in the measurement label.

            Returns:
                None
            """
            if len(self.selected_points) < 2:
                self.measurement_label.setText("Select at least two points to measure.")
                return

            distances = self.calculate_distances(self.selected_points)
            if len(self.selected_points) > 2:
                area = self.calculate_area(self.selected_points)
                self.measurement_label.setText(f"Distances: {distances}\nArea: {area:.2f} square meters")
            else:
                self.measurement_label.setText(f"Distances: {distances}")

    def calculate_distances(self, points):
        """Calculates the distances between consecutive points using the Haversine formula, limited to the latest 5 segments.

        Args:
            points (list): A list of points, where each point is represented as a tuple of (longitude, latitude).

        Returns:
            list: A list of distances between consecutive points, rounded to 2 decimal places.
        """
        # Limit points to the last 5 points or fewer if there are not 5 points
        points = points[-6:]  # Get the last 6 points to calculate 5 segments
        distances = []
        for i in range(1, len(points)):
            lon1, lat1 = points[i - 1]
            lon2, lat2 = points[i]
            distance = self.haversine(lat1, lon1, lat2, lon2)
            distances.append(round(distance, 2))

        return distances

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculates the Haversine distance between two points on the Earth.

        Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

        Returns:
        float: The Haversine distance between the two points in kilometers.
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = EARTH_RADIUS * c
        return distance

    def calculate_area(self, points):
        """
        Calculates the area of the polygon formed by the points in square meters.

        Args:
            points (list): A list of tuples representing the latitude and longitude of each point.

        Returns:
            float: The area of the polygon in square meters.
        """
        if len(points) < 3:
            return 0.0

        # Convert latitude and longitude to Cartesian coordinates
        R = EARTH_RADIUS
        x = []
        y = []

        for lon, lat in points:
            lon_rad = np.radians(lon)
            lat_rad = np.radians(lat)
            x.append(R * lon_rad * np.cos(lat_rad))
            y.append(R * lat_rad)

        # Apply Shoelace formula to the Cartesian coordinates
        x = np.array(x)
        y = np.array(y)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area


    def plot_coordinates(self):
        """
        Plots the loaded coordinates on the canvas.

        This method plots the longitude and latitude coordinates on a canvas using matplotlib.
        It first checks if the data is valid by calling the `validate_data` method.
        If the data is valid, it creates a subplot on the figure, clears the subplot, and plots the coordinates.
        It sets the title, x-label, and y-label of the plot.
        Finally, it draws the canvas and connects a button press event to the `on_click` method.

        Returns:
            None
        """
        if not self.validate_data():
            return
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(self.longitudes, self.latitudes, linestyle='-', color='b')
        ax.set_title("Drone Flight Path")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def reset_plot(self):
        """Clears the plot and resets selected points and labels.

        This method clears the plot by removing all selected points and labels.
        It also updates the coordinates and measurement labels to indicate that no data is loaded.
        Finally, it clears the matplotlib figure and redraws the canvas to reflect the cleared plot.
        """
        self.selected_points.clear()  # Clear the list of selected points
        self.coordinates_label.setText("No coordinates loaded.")
        self.measurement_label.setText("No measurements taken.")
        self.figure.clear()  # Clear the matplotlib figure
        self.canvas.draw()  # Redraw the canvas to reflect the cleared plot

    def validate_data(self):
            """Validates if the data is loaded for plotting.

            Returns:
                bool: True if data is available, False otherwise.
            """
            if not self.latitudes or not self.longitudes:
                self.coordinates_label.setText("No data available to plot.")
                return False
            return True

    # File operation methods
    def load_file(self):
        """Loads flight record from a CSV file.

        This method opens a file dialog to allow the user to select a CSV file containing flight records.
        It reads the CSV file using pandas and extracts latitude and longitude information from the 'OSD.latitude'
        and 'OSD.longitude' columns. The extracted coordinates are stored in the 'latitudes' and 'longitudes' lists.
        The method also updates the 'coordinates_label' widget to display information about the loaded coordinates.

        Raises:
            KeyError: If the 'OSD.latitude' or 'OSD.longitude' columns are not found in the CSV file.
            Exception: If any other error occurs during the file loading process.

        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Flight Record", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            try:
                df = pd.read_csv(file_name, sep=',', engine='python', skiprows=1)
                df.columns = df.columns.str.strip()
                self.latitudes = df['OSD.latitude'].dropna().tolist()
                self.longitudes = df['OSD.longitude'].dropna().tolist()
                lat_long_info = f"Coordinates loaded: {len(self.latitudes)}\n"
                if self.latitudes:
                    lat_long_info += f"First Latitude: {self.latitudes[0]}, First Longitude: {self.longitudes[0]}"
                else:
                    lat_long_info += "No coordinates available."
                self.coordinates_label.setText(lat_long_info)
            except KeyError as e:
                self.coordinates_label.setText(f"KeyError: {e}")
                self.latitudes = []
                self.longitudes = []
            except Exception as e:
                self.coordinates_label.setText(f"An error occurred: {e}")
                self.latitudes = []
                self.longitudes = []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
