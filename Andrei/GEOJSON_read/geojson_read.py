import geopandas as gpd
import matplotlib.pyplot as plt

df = gpd.read_file("file.geojson")
map = df.explore()
map.save("map.html")