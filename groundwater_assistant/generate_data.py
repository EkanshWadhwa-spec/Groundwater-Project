import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Comprehensive Groundwater Monitoring Sites (India CGWB + Global USGS NWIS)
# Covering major Indian states & districts along with international sites
sites_data = [
    # ----------------- INDIA: PUNJAB -----------------
    {
        "site_no": "CGWB-PB-LUD01",
        "station_nm": "CGWB Central Piezometer Station Ludhiana",
        "country": "India",
        "state": "Punjab",
        "state_code": "PB",
        "county": "Ludhiana",
        "dec_lat_va": 30.9010,
        "dec_long_va": 75.8573,
        "altitude_va": 800.0,
        "well_depth_va": 380.0,
        "aquifer": "Indo-Gangetic Alluvial Aquifer System",
        "current_water_level_ft": 138.5,
        "surface_soil_moisture_pct": 28.4,
        "rootzone_soil_moisture_pct": 32.1,
        "soil_moisture_condition": "Optimal Agricultural Moisture",
        "soil_type": "Alluvial Silt Loam",
        "water_scarcity_index": "Over-Exploited (Critical Overdraft)",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-PB-AMR02",
        "station_nm": "Amritsar Border Belt Groundwater Station",
        "country": "India",
        "state": "Punjab",
        "state_code": "PB",
        "county": "Amritsar",
        "dec_lat_va": 31.6340,
        "dec_long_va": 74.8723,
        "altitude_va": 760.0,
        "well_depth_va": 320.0,
        "aquifer": "Indo-Gangetic Alluvial Aquifer System",
        "current_water_level_ft": 112.4,
        "surface_soil_moisture_pct": 27.0,
        "rootzone_soil_moisture_pct": 30.5,
        "soil_moisture_condition": "Optimal Moisture",
        "soil_type": "Alluvial Loam",
        "water_scarcity_index": "Over-Exploited",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: HARYANA -----------------
    {
        "site_no": "CGWB-HR-KAR01",
        "station_nm": "Karnal Agricultural Basin Monitoring Well",
        "country": "India",
        "state": "Haryana",
        "state_code": "HR",
        "county": "Karnal",
        "dec_lat_va": 29.6857,
        "dec_long_va": 76.9905,
        "altitude_va": 820.0,
        "well_depth_va": 350.0,
        "aquifer": "Indo-Gangetic Alluvial Aquifer System",
        "current_water_level_ft": 124.6,
        "surface_soil_moisture_pct": 25.8,
        "rootzone_soil_moisture_pct": 29.2,
        "soil_moisture_condition": "Adequate Moisture",
        "soil_type": "Sandy Clay Loam",
        "water_scarcity_index": "Over-Exploited",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-HR-GUR02",
        "station_nm": "Gurugram Cyber City Deep Piezometer",
        "country": "India",
        "state": "Haryana",
        "state_code": "HR",
        "county": "Gurugram",
        "dec_lat_va": 28.4595,
        "dec_long_va": 77.0266,
        "altitude_va": 745.0,
        "well_depth_va": 450.0,
        "aquifer": "Delhi Quartzite & Alluvium System",
        "current_water_level_ft": 148.2,
        "surface_soil_moisture_pct": 14.2,
        "rootzone_soil_moisture_pct": 17.5,
        "soil_moisture_condition": "Dry / Urban Deficit",
        "soil_type": "Silty Sand",
        "water_scarcity_index": "Critical Urban Overdraft",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: DELHI (NCR) -----------------
    {
        "site_no": "CGWB-DL-DEL01",
        "station_nm": "National Capital Region Central Station New Delhi",
        "country": "India",
        "state": "Delhi",
        "state_code": "DL",
        "county": "New Delhi",
        "dec_lat_va": 28.6139,
        "dec_long_va": 77.2090,
        "altitude_va": 700.0,
        "well_depth_va": 400.0,
        "aquifer": "Yamuna Floodplain Alluvium & Ridge Quartzite",
        "current_water_level_ft": 118.0,
        "surface_soil_moisture_pct": 16.5,
        "rootzone_soil_moisture_pct": 19.8,
        "soil_moisture_condition": "Moderately Dry",
        "soil_type": "Yamuna Alluvial Sand",
        "water_scarcity_index": "Over-Exploited / High Urban Stress",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: RAJASTHAN -----------------
    {
        "site_no": "CGWB-RJ-JOD01",
        "station_nm": "Jodhpur Marwar Arid Zone Observatory",
        "country": "India",
        "state": "Rajasthan",
        "state_code": "RJ",
        "county": "Jodhpur",
        "dec_lat_va": 26.2389,
        "dec_long_va": 73.0243,
        "altitude_va": 758.0,
        "well_depth_va": 550.0,
        "aquifer": "Jodhpur Sandstone & Fractured Rock Aquifer",
        "current_water_level_ft": 218.4,
        "surface_soil_moisture_pct": 7.8,
        "rootzone_soil_moisture_pct": 10.2,
        "soil_moisture_condition": "Severely Dry / Desert Drought Warning",
        "soil_type": "Desert Sand & Calcareous",
        "water_scarcity_index": "Critical / Extreme Depletion",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-RJ-JAI02",
        "station_nm": "Jaipur Ramgarh Basin Sentinel Station",
        "country": "India",
        "state": "Rajasthan",
        "state_code": "RJ",
        "county": "Jaipur",
        "dec_lat_va": 26.9124,
        "dec_long_va": 75.7873,
        "altitude_va": 1410.0,
        "well_depth_va": 480.0,
        "aquifer": "Quaternary Alluvium & Aravalli Quartzite",
        "current_water_level_ft": 165.7,
        "surface_soil_moisture_pct": 12.5,
        "rootzone_soil_moisture_pct": 15.3,
        "soil_moisture_condition": "Arid / Deficit",
        "soil_type": "Sandy Loam",
        "water_scarcity_index": "Over-Exploited",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-RJ-JAI03",
        "station_nm": "Jaisalmer Thar Deep Piezometer",
        "country": "India",
        "state": "Rajasthan",
        "state_code": "RJ",
        "county": "Jaisalmer",
        "dec_lat_va": 26.9157,
        "dec_long_va": 70.9083,
        "altitude_va": 738.0,
        "well_depth_va": 650.0,
        "aquifer": "Lathi Sandstone Deep Aquifer",
        "current_water_level_ft": 264.0,
        "surface_soil_moisture_pct": 6.2,
        "rootzone_soil_moisture_pct": 8.5,
        "soil_moisture_condition": "Hyper-Arid / Severe Desert Deficit",
        "soil_type": "Aeolian Sand",
        "water_scarcity_index": "Critical Depletion",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: UTTAR PRADESH -----------------
    {
        "site_no": "CGWB-UP-VAR01",
        "station_nm": "Varanasi Gangetic Basin Monitoring Well",
        "country": "India",
        "state": "Uttar Pradesh",
        "state_code": "UP",
        "county": "Varanasi",
        "dec_lat_va": 25.3176,
        "dec_long_va": 82.9739,
        "altitude_va": 265.0,
        "well_depth_va": 260.0,
        "aquifer": "Central Gangetic Plain Alluvium",
        "current_water_level_ft": 38.5,
        "surface_soil_moisture_pct": 32.6,
        "rootzone_soil_moisture_pct": 36.4,
        "soil_moisture_condition": "Optimal Agricultural Moisture",
        "soil_type": "Fertile Silt Loam",
        "water_scarcity_index": "Safe / Sustainable Recharge",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-UP-LUC02",
        "station_nm": "Lucknow Gomti Basin Piezometer",
        "country": "India",
        "state": "Uttar Pradesh",
        "state_code": "UP",
        "county": "Lucknow",
        "dec_lat_va": 26.8467,
        "dec_long_va": 80.9462,
        "altitude_va": 403.0,
        "well_depth_va": 310.0,
        "aquifer": "Central Gangetic Plain Alluvium",
        "current_water_level_ft": 48.2,
        "surface_soil_moisture_pct": 29.8,
        "rootzone_soil_moisture_pct": 33.2,
        "soil_moisture_condition": "Adequate Moisture",
        "soil_type": "Alluvial Loam",
        "water_scarcity_index": "Semi-Critical",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-UP-AGR03",
        "station_nm": "Agra Yamuna Riverine Observation Well",
        "country": "India",
        "state": "Uttar Pradesh",
        "state_code": "UP",
        "county": "Agra",
        "dec_lat_va": 27.1767,
        "dec_long_va": 78.0081,
        "altitude_va": 554.0,
        "well_depth_va": 340.0,
        "aquifer": "Yamuna Alluvium & Vindhyan Sandstone",
        "current_water_level_ft": 62.4,
        "surface_soil_moisture_pct": 21.5,
        "rootzone_soil_moisture_pct": 24.8,
        "soil_moisture_condition": "Moderate Moisture",
        "soil_type": "Sandy Silt",
        "water_scarcity_index": "Semi-Critical / Saline Intrusion",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: MAHARASHTRA -----------------
    {
        "site_no": "CGWB-MH-PUN01",
        "station_nm": "Pune Deccan Plateau Monitoring Station",
        "country": "India",
        "state": "Maharashtra",
        "state_code": "MH",
        "county": "Pune",
        "dec_lat_va": 18.5204,
        "dec_long_va": 73.8567,
        "altitude_va": 1840.0,
        "well_depth_va": 310.0,
        "aquifer": "Deccan Traps Basaltic Hard Rock Aquifer",
        "current_water_level_ft": 32.8,
        "surface_soil_moisture_pct": 24.5,
        "rootzone_soil_moisture_pct": 31.0,
        "soil_moisture_condition": "Adequate Moisture",
        "soil_type": "Black Cotton Soil (Regur)",
        "water_scarcity_index": "Semi-Critical (Seasonal Hard Rock)",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-MH-SOL02",
        "station_nm": "Solapur Marathwada Drought Sentinel Station",
        "country": "India",
        "state": "Maharashtra",
        "state_code": "MH",
        "county": "Solapur",
        "dec_lat_va": 17.6599,
        "dec_long_va": 75.9064,
        "altitude_va": 1500.0,
        "well_depth_va": 380.0,
        "aquifer": "Deccan Traps Basaltic Hard Rock Aquifer",
        "current_water_level_ft": 74.2,
        "surface_soil_moisture_pct": 14.8,
        "rootzone_soil_moisture_pct": 18.2,
        "soil_moisture_condition": "Dry / Drought Vulnerable",
        "soil_type": "Medium Black Soil",
        "water_scarcity_index": "Over-Exploited / Drought Prone",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-MH-NAG03",
        "station_nm": "Nagpur Vidarbha Agro-Hydrological Well",
        "country": "India",
        "state": "Maharashtra",
        "state_code": "MH",
        "county": "Nagpur",
        "dec_lat_va": 21.1458,
        "dec_long_va": 79.0882,
        "altitude_va": 1020.0,
        "well_depth_va": 290.0,
        "aquifer": "Deccan Basalt & Gondwana Sandstone",
        "current_water_level_ft": 41.6,
        "surface_soil_moisture_pct": 26.4,
        "rootzone_soil_moisture_pct": 30.5,
        "soil_moisture_condition": "Optimal Moisture",
        "soil_type": "Deep Black Cotton Soil",
        "water_scarcity_index": "Semi-Critical",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: GUJARAT -----------------
    {
        "site_no": "CGWB-GJ-AHM01",
        "station_nm": "Ahmedabad Sabarmati Alluvial Well",
        "country": "India",
        "state": "Gujarat",
        "state_code": "GJ",
        "county": "Ahmedabad",
        "dec_lat_va": 23.0225,
        "dec_long_va": 72.5714,
        "altitude_va": 174.0,
        "well_depth_va": 450.0,
        "aquifer": "North Gujarat Alluvial & Mehsana Basin",
        "current_water_level_ft": 156.4,
        "surface_soil_moisture_pct": 15.6,
        "rootzone_soil_moisture_pct": 19.2,
        "soil_moisture_condition": "Dry / Semi-Arid",
        "soil_type": "Sandy Clay Loam",
        "water_scarcity_index": "Over-Exploited / High Overdraft",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-GJ-RAJ02",
        "station_nm": "Rajkot Saurashtra Peninsula Station",
        "country": "India",
        "state": "Gujarat",
        "state_code": "GJ",
        "county": "Rajkot",
        "dec_lat_va": 22.3039,
        "dec_long_va": 70.8022,
        "altitude_va": 453.0,
        "well_depth_va": 320.0,
        "aquifer": "Deccan Basaltic & Mesozoic Sandstone",
        "current_water_level_ft": 68.3,
        "surface_soil_moisture_pct": 17.2,
        "rootzone_soil_moisture_pct": 21.0,
        "soil_moisture_condition": "Moderate-Dry",
        "soil_type": "Medium Black Soil",
        "water_scarcity_index": "Semi-Critical",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: KARNATAKA -----------------
    {
        "site_no": "CGWB-KA-BLR01",
        "station_nm": "Bengaluru Urban Granitic Piezometer",
        "country": "India",
        "state": "Karnataka",
        "state_code": "KA",
        "county": "Bengaluru",
        "dec_lat_va": 12.9716,
        "dec_long_va": 77.5946,
        "altitude_va": 3020.0,
        "well_depth_va": 650.0,
        "aquifer": "Peninsular Gneiss & Granitic Hard Rock Aquifer",
        "current_water_level_ft": 142.0,
        "surface_soil_moisture_pct": 18.2,
        "rootzone_soil_moisture_pct": 22.4,
        "soil_moisture_condition": "Moderately Dry / High Urban Drawdown",
        "soil_type": "Red Sandy Loam",
        "water_scarcity_index": "Over-Exploited",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-KA-MYS02",
        "station_nm": "Mysuru Cauvery Basin Groundwater Station",
        "country": "India",
        "state": "Karnataka",
        "state_code": "KA",
        "county": "Mysuru",
        "dec_lat_va": 12.2958,
        "dec_long_va": 76.6394,
        "altitude_va": 2520.0,
        "well_depth_va": 340.0,
        "aquifer": "Peninsular Gneissic Complex",
        "current_water_level_ft": 44.8,
        "surface_soil_moisture_pct": 26.5,
        "rootzone_soil_moisture_pct": 31.0,
        "soil_moisture_condition": "Optimal Moisture",
        "soil_type": "Red Loam",
        "water_scarcity_index": "Safe / Sustainable",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: TAMIL NADU -----------------
    {
        "site_no": "CGWB-TN-CHE01",
        "station_nm": "Chennai Coastal Aquifer Sentinel Well",
        "country": "India",
        "state": "Tamil Nadu",
        "state_code": "TN",
        "county": "Chennai",
        "dec_lat_va": 13.0827,
        "dec_long_va": 80.2707,
        "altitude_va": 22.0,
        "well_depth_va": 210.0,
        "aquifer": "Coromandel Coastal Alluvium & Weathered Rock",
        "current_water_level_ft": 28.4,
        "surface_soil_moisture_pct": 31.2,
        "rootzone_soil_moisture_pct": 35.8,
        "soil_moisture_condition": "Adequate / Coastal Saturation",
        "soil_type": "Coastal Alluvial Sand",
        "water_scarcity_index": "Over-Exploited / Saline Intrusion Alert",
        "status": "Active CGWB Monitoring"
    },
    {
        "site_no": "CGWB-TN-COI02",
        "station_nm": "Coimbatore Noyyal Basin Deep Station",
        "country": "India",
        "state": "Tamil Nadu",
        "state_code": "TN",
        "county": "Coimbatore",
        "dec_lat_va": 11.0168,
        "dec_long_va": 76.9558,
        "altitude_va": 1350.0,
        "well_depth_va": 520.0,
        "aquifer": "Southern Crystalline Charnockite Hard Rock",
        "current_water_level_ft": 108.5,
        "surface_soil_moisture_pct": 19.0,
        "rootzone_soil_moisture_pct": 23.4,
        "soil_moisture_condition": "Moderately Dry",
        "soil_type": "Red Loamy Clay",
        "water_scarcity_index": "Over-Exploited",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: TELANGANA -----------------
    {
        "site_no": "CGWB-TS-HYD01",
        "station_nm": "Hyderabad Deccan Granitic Station",
        "country": "India",
        "state": "Telangana",
        "state_code": "TS",
        "county": "Hyderabad",
        "dec_lat_va": 17.3850,
        "dec_long_va": 78.4867,
        "altitude_va": 1778.0,
        "well_depth_va": 420.0,
        "aquifer": "Granite & Fractured Gneiss Hard Rock",
        "current_water_level_ft": 52.6,
        "surface_soil_moisture_pct": 22.4,
        "rootzone_soil_moisture_pct": 26.8,
        "soil_moisture_condition": "Adequate Moisture",
        "soil_type": "Red Chalky Loam",
        "water_scarcity_index": "Semi-Critical",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: WEST BENGAL -----------------
    {
        "site_no": "CGWB-WB-KOL01",
        "station_nm": "Kolkata Lower Ganga Delta Piezometer",
        "country": "India",
        "state": "West Bengal",
        "state_code": "WB",
        "county": "Kolkata",
        "dec_lat_va": 22.5726,
        "dec_long_va": 88.3639,
        "altitude_va": 30.0,
        "well_depth_va": 280.0,
        "aquifer": "Bengal Delta Deltaic Alluvium",
        "current_water_level_ft": 16.2,
        "surface_soil_moisture_pct": 42.0,
        "rootzone_soil_moisture_pct": 46.5,
        "soil_moisture_condition": "High Moisture / Near Saturation",
        "soil_type": "Deltaic Silty Clay",
        "water_scarcity_index": "Safe / Shallow Water Table",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: BIHAR -----------------
    {
        "site_no": "CGWB-BR-PAT01",
        "station_nm": "Patna Gangetic Alluvial Station",
        "country": "India",
        "state": "Bihar",
        "state_code": "BR",
        "county": "Patna",
        "dec_lat_va": 25.5941,
        "dec_long_va": 85.1376,
        "altitude_va": 174.0,
        "well_depth_va": 240.0,
        "aquifer": "Middle Gangetic Plain Alluvium",
        "current_water_level_ft": 24.8,
        "surface_soil_moisture_pct": 35.4,
        "rootzone_soil_moisture_pct": 39.2,
        "soil_moisture_condition": "Optimal Agricultural Moisture",
        "soil_type": "Fertile Gangetic Alluvial Silt",
        "water_scarcity_index": "Safe / Abundant Aquifer",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: KERALA -----------------
    {
        "site_no": "CGWB-KL-ERN01",
        "station_nm": "Ernakulam Kochi Coastal Observatory",
        "country": "India",
        "state": "Kerala",
        "state_code": "KL",
        "county": "Ernakulam",
        "dec_lat_va": 9.9816,
        "dec_long_va": 76.2999,
        "altitude_va": 15.0,
        "well_depth_va": 180.0,
        "aquifer": "Coastal Alluvium & Western Ghats Laterite",
        "current_water_level_ft": 14.5,
        "surface_soil_moisture_pct": 43.8,
        "rootzone_soil_moisture_pct": 48.0,
        "soil_moisture_condition": "High Moisture / Saturated",
        "soil_type": "Coastal Laterite & Sand",
        "water_scarcity_index": "Safe / Abundant High Rainfall",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: MADHYA PRADESH -----------------
    {
        "site_no": "CGWB-MP-IND01",
        "station_nm": "Indore Malwa Plateau Groundwater Station",
        "country": "India",
        "state": "Madhya Pradesh",
        "state_code": "MP",
        "county": "Indore",
        "dec_lat_va": 22.7196,
        "dec_long_va": 75.8577,
        "altitude_va": 1814.0,
        "well_depth_va": 360.0,
        "aquifer": "Malwa Deccan Traps Basalt",
        "current_water_level_ft": 58.7,
        "surface_soil_moisture_pct": 23.2,
        "rootzone_soil_moisture_pct": 27.5,
        "soil_moisture_condition": "Adequate Moisture",
        "soil_type": "Deep Black Soil",
        "water_scarcity_index": "Semi-Critical",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INDIA: ASSAM -----------------
    {
        "site_no": "CGWB-AS-GUW01",
        "station_nm": "Guwahati Brahmaputra Alluvial Well",
        "country": "India",
        "state": "Assam",
        "state_code": "AS",
        "county": "Kamrup",
        "dec_lat_va": 26.1445,
        "dec_long_va": 91.7362,
        "altitude_va": 180.0,
        "well_depth_va": 190.0,
        "aquifer": "Brahmaputra Valley Alluvial Aquifer",
        "current_water_level_ft": 12.8,
        "surface_soil_moisture_pct": 46.5,
        "rootzone_soil_moisture_pct": 49.2,
        "soil_moisture_condition": "Saturated Moisture / High Water Table",
        "soil_type": "Riverine Silt & Clay",
        "water_scarcity_index": "Safe / Very High Recharge",
        "status": "Active CGWB Monitoring"
    },

    # ----------------- INTERNATIONAL (USGS NWIS) -----------------
    {
        "site_no": "342500118250001",
        "station_nm": "Central Valley Aquifer Well 14-B",
        "country": "United States",
        "state": "California",
        "state_code": "CA",
        "county": "Kern",
        "dec_lat_va": 35.3733,
        "dec_long_va": -119.0187,
        "altitude_va": 405.0,
        "well_depth_va": 450.0,
        "aquifer": "California Central Valley Aquifer System",
        "current_water_level_ft": 142.8,
        "surface_soil_moisture_pct": 11.2,
        "rootzone_soil_moisture_pct": 14.8,
        "soil_moisture_condition": "Severely Dry / Drought Warning",
        "soil_type": "Sandy Loam",
        "water_scarcity_index": "Severe Overdraft / High Depletion Risk",
        "status": "Active USGS Monitoring"
    },
    {
        "site_no": "293830098255501",
        "station_nm": "J-17 Edwards Aquifer Index Well",
        "country": "United States",
        "state": "Texas",
        "state_code": "TX",
        "county": "Bexar",
        "dec_lat_va": 29.4589,
        "dec_long_va": -98.4481,
        "altitude_va": 730.0,
        "well_depth_va": 520.0,
        "aquifer": "Edwards-Trinity Aquifer System",
        "current_water_level_ft": 65.4,
        "surface_soil_moisture_pct": 19.4,
        "rootzone_soil_moisture_pct": 23.5,
        "soil_moisture_condition": "Moderately Dry / Mild Deficit",
        "soil_type": "Clay Loam",
        "water_scarcity_index": "Moderate Stress (Karst Sensitive)",
        "status": "Active USGS Monitoring"
    },
    {
        "site_no": "280255082260501",
        "station_nm": "Floridan Aquifer Coastal Sentinel Well",
        "country": "United States",
        "state": "Florida",
        "state_code": "FL",
        "county": "Marion",
        "dec_lat_va": 29.1872,
        "dec_long_va": -82.1401,
        "altitude_va": 85.0,
        "well_depth_va": 160.0,
        "aquifer": "Floridan Aquifer System",
        "current_water_level_ft": 18.6,
        "surface_soil_moisture_pct": 39.5,
        "rootzone_soil_moisture_pct": 44.2,
        "soil_moisture_condition": "High Moisture / Near Saturation",
        "soil_type": "Sandy Clay",
        "water_scarcity_index": "Sustainable Recharge",
        "status": "Active USGS Monitoring"
    },
    {
        "site_no": "332450111584001",
        "station_nm": "Phoenix Basin Alluvial Well 3",
        "country": "United States",
        "state": "Arizona",
        "state_code": "AZ",
        "county": "Maricopa",
        "dec_lat_va": 33.4484,
        "dec_long_va": -112.0740,
        "altitude_va": 1086.0,
        "well_depth_va": 600.0,
        "aquifer": "Basin and Range Basin-Fill Aquifers",
        "current_water_level_ft": 215.1,
        "surface_soil_moisture_pct": 8.7,
        "rootzone_soil_moisture_pct": 11.3,
        "soil_moisture_condition": "Extremely Arid / Severe Deficit",
        "soil_type": "Gravelly Sand",
        "water_scarcity_index": "Critical Deep Overdraft",
        "status": "Active USGS Monitoring"
    },
    {
        "site_no": "405012100123401",
        "station_nm": "High Plains Ogallala Test Station 8",
        "country": "United States",
        "state": "Nebraska",
        "state_code": "NE",
        "county": "Buffalo",
        "dec_lat_va": 40.6995,
        "dec_long_va": -99.0818,
        "altitude_va": 2150.0,
        "well_depth_va": 280.0,
        "aquifer": "High Plains (Ogallala) Aquifer",
        "current_water_level_ft": 88.2,
        "surface_soil_moisture_pct": 26.2,
        "rootzone_soil_moisture_pct": 29.8,
        "soil_moisture_condition": "Adequate Moisture",
        "soil_type": "Fine Silt Loam",
        "water_scarcity_index": "Moderate Agricultural Drawdown",
        "status": "Active USGS Monitoring"
    },
    {
        "site_no": "01646500",
        "station_nm": "USGS Potom-Shenandoah Monitoring Well 1",
        "country": "United States",
        "state": "Maryland",
        "state_code": "MD",
        "county": "Montgomery",
        "dec_lat_va": 39.1245,
        "dec_long_va": -77.2189,
        "altitude_va": 320.0,
        "well_depth_va": 185.0,
        "aquifer": "Piedmont and Blue Ridge Crystalline-Rock Aquifers",
        "current_water_level_ft": 24.3,
        "surface_soil_moisture_pct": 31.5,
        "rootzone_soil_moisture_pct": 36.8,
        "soil_moisture_condition": "Optimal Moisture for Agriculture",
        "soil_type": "Silty Loam",
        "water_scarcity_index": "Sustainable Recharge",
        "status": "Active USGS Monitoring"
    }
]

df_sites = pd.DataFrame(sites_data)
sites_csv_path = os.path.join(DATA_DIR, "usgs_groundwater_sites.csv")
df_sites.to_csv(sites_csv_path, index=False)
print(f"Saved {len(df_sites)} comprehensive sites to {sites_csv_path}")

# 2. Time-Series Water Levels (Monthly data for past 24 months per site)
np.random.seed(42)
random.seed(42)

timeseries_rows = []
base_date = datetime(2024, 11, 1)

for site in sites_data:
    site_no = site["site_no"]
    base_depth = site["current_water_level_ft"]
    altitude = site["altitude_va"]
    is_india = site["country"] == "India"
    
    # In India: strong monsoon recharge in July-Sept (months 7, 8, 9) decreases depth (water table rises)
    # followed by heavy pre-monsoon drawdown in March-June (months 3, 4, 5, 6)
    trend_slope = 0.20 if "Over-Exploited" in site["water_scarcity_index"] else -0.05 if "Safe" in site["water_scarcity_index"] else 0.08
    
    for month_offset in range(24, -1, -1):
        dt = base_date - timedelta(days=month_offset * 30.5)
        m = dt.month
        
        if is_india:
            # Monsoon cycle
            if m in [7, 8, 9]:
                seasonal = -7.5 * (1.0 if m == 8 else 0.7) # Monsoon recharge: water table rises
            elif m in [4, 5, 6]:
                seasonal = 5.2 # Summer heat & pre-monsoon drawdown
            else:
                seasonal = 0.5
        else:
            seasonal = 3.5 * np.sin(2 * np.pi * m / 12)
            
        noise = np.random.normal(0, 0.4)
        depth = round(base_depth - (trend_slope * month_offset) + seasonal + noise, 2)
        depth = max(4.0, depth)
        elev = round(altitude - depth, 2)
        
        timeseries_rows.append({
            "site_no": site_no,
            "station_nm": site["station_nm"],
            "record_date": dt.strftime("%Y-%m-%d"),
            "water_level_depth_ft": depth,
            "water_level_elevation_ft": elev,
            "approval_status": "Approved (CGWB/USGS Verified)" if month_offset > 2 else "Provisional"
        })

df_ts = pd.DataFrame(timeseries_rows)
ts_csv_path = os.path.join(DATA_DIR, "usgs_water_levels_timeseries.csv")
df_ts.to_csv(ts_csv_path, index=False)
print(f"Saved {len(df_ts)} timeseries records to {ts_csv_path}")

# 3. Water Quality Dataset (pH, TDS, Nitrate, Fluoride, Potability Standards)
# Aligned with BIS IS 10500:2012 (India) & EPA (US)
quality_profiles = {
    # Punjab: Fertilizer runoff, elevated nitrate
    "CGWB-PB-LUD01": {"ph": 7.4, "cond": 890, "tds": 560, "nitrate": 18.5, "temp": 22.0, "status": "Elevated Nitrate (Fertilizer Runoff, Treatment Advised)", "pot": "Warning (High Nitrate)"},
    "CGWB-PB-AMR02": {"ph": 7.6, "cond": 780, "tds": 490, "nitrate": 12.0, "temp": 21.5, "status": "Acceptable Potability / Mild Hardness", "pot": "Compliant"},
    # Haryana: Saline & nitrate in parts
    "CGWB-HR-KAR01": {"ph": 7.8, "cond": 920, "tds": 580, "nitrate": 14.2, "temp": 23.0, "status": "Elevated Hardness and Nitrate", "pot": "Requires Filtration"},
    "CGWB-HR-GUR02": {"ph": 8.0, "cond": 1350, "tds": 840, "nitrate": 9.5, "temp": 24.2, "status": "High Total Dissolved Solids (Hard Water)", "pot": "High Mineralization"},
    # Delhi: Urban groundwater
    "CGWB-DL-DEL01": {"ph": 7.7, "cond": 1450, "tds": 910, "nitrate": 22.0, "temp": 24.5, "status": "Elevated Nitrate & High TDS (Urban Leaching)", "pot": "Non-Compliant / Warning"},
    # Rajasthan: High TDS, Fluoride in desert aquifers
    "CGWB-RJ-JOD01": {"ph": 8.3, "cond": 2100, "tds": 1420, "nitrate": 16.5, "temp": 27.5, "status": "High Salinity & Brackish Ground Water", "pot": "Non-Compliant (High TDS)"},
    "CGWB-RJ-JAI02": {"ph": 7.9, "cond": 1280, "tds": 790, "nitrate": 11.0, "temp": 26.0, "status": "Moderate Salinity, Requires RO", "pot": "Requires Treatment"},
    "CGWB-RJ-JAI03": {"ph": 8.4, "cond": 2400, "tds": 1680, "nitrate": 8.2, "temp": 29.0, "status": "Deep Fossil Saline Water", "pot": "Non-Compliant (Brackish)"},
    # UP: Gangetic alluvium, generally good
    "CGWB-UP-VAR01": {"ph": 7.3, "cond": 480, "tds": 310, "nitrate": 3.8, "temp": 24.0, "status": "Pristine Gangetic Groundwater, Safe & Potable", "pot": "Compliant (Safe for Drinking)"},
    "CGWB-UP-LUC02": {"ph": 7.5, "cond": 540, "tds": 345, "nitrate": 5.2, "temp": 24.5, "status": "Good Quality, Potable", "pot": "Compliant (Safe for Drinking)"},
    "CGWB-UP-AGR03": {"ph": 7.8, "cond": 1100, "tds": 690, "nitrate": 8.4, "temp": 25.2, "status": "Moderate Hardness", "pot": "Acceptable / Mild Hardness"},
    # Maharashtra: Deccan Basalt
    "CGWB-MH-PUN01": {"ph": 7.4, "cond": 520, "tds": 330, "nitrate": 4.1, "temp": 23.8, "status": "High Purity Basaltic Aquifer Water", "pot": "Compliant (Safe for Drinking)"},
    "CGWB-MH-SOL02": {"ph": 7.9, "cond": 1200, "tds": 760, "nitrate": 12.8, "temp": 26.5, "status": "Moderate Saline Concentration", "pot": "Requires Treatment"},
    "CGWB-MH-NAG03": {"ph": 7.5, "cond": 610, "tds": 390, "nitrate": 6.0, "temp": 25.0, "status": "Standard Potable Quality", "pot": "Compliant (Safe for Drinking)"},
    # Gujarat
    "CGWB-GJ-AHM01": {"ph": 8.1, "cond": 1380, "tds": 860, "nitrate": 10.5, "temp": 26.2, "status": "Elevated Dissolved Solids", "pot": "Requires Treatment"},
    "CGWB-GJ-RAJ02": {"ph": 7.8, "cond": 950, "tds": 610, "nitrate": 7.2, "temp": 25.8, "status": "Mildly Hard Water", "pot": "Acceptable"},
    # Karnataka
    "CGWB-KA-BLR01": {"ph": 7.6, "cond": 890, "tds": 570, "nitrate": 15.4, "temp": 22.8, "status": "Elevated Nitrate from Urban Runoff", "pot": "Warning (High Nitrate)"},
    "CGWB-KA-MYS02": {"ph": 7.2, "cond": 420, "tds": 270, "nitrate": 2.8, "temp": 23.2, "status": "Pristine Cauvery Basin Water", "pot": "Compliant (Safe for Drinking)"},
    # Tamil Nadu
    "CGWB-TN-CHE01": {"ph": 7.9, "cond": 1600, "tds": 980, "nitrate": 8.0, "temp": 27.0, "status": "High Electrical Conductance / Coastal Saline", "pot": "Coastal Salinity Warning"},
    "CGWB-TN-COI02": {"ph": 7.7, "cond": 910, "tds": 580, "nitrate": 11.2, "temp": 25.4, "status": "Elevated Hardness", "pot": "Requires Treatment"},
    # Telangana
    "CGWB-TS-HYD01": {"ph": 7.5, "cond": 740, "tds": 470, "nitrate": 9.1, "temp": 25.5, "status": "Standard Municipal Potability", "pot": "Compliant (Safe for Drinking)"},
    # West Bengal: Shallow alluvium
    "CGWB-WB-KOL01": {"ph": 7.3, "cond": 650, "tds": 410, "nitrate": 2.1, "temp": 26.0, "status": "Fresh Deltaic Water (Arsenic Tested Safe)", "pot": "Compliant (Safe for Drinking)"},
    # Bihar
    "CGWB-BR-PAT01": {"ph": 7.4, "cond": 490, "tds": 315, "nitrate": 3.2, "temp": 24.8, "status": "Abundant Potable Alluvial Water", "pot": "Compliant (Safe for Drinking)"},
    # Kerala: High rainfall
    "CGWB-KL-ERN01": {"ph": 6.9, "cond": 280, "tds": 175, "nitrate": 1.1, "temp": 26.5, "status": "Pristine Natural Lateritic Spring Quality", "pot": "Compliant (Safe for Drinking)"},
    # Madhya Pradesh
    "CGWB-MP-IND01": {"ph": 7.5, "cond": 630, "tds": 405, "nitrate": 5.8, "temp": 24.2, "status": "Good Quality Drinking Water", "pot": "Compliant (Safe for Drinking)"},
    # Assam
    "CGWB-AS-GUW01": {"ph": 7.1, "cond": 310, "tds": 195, "nitrate": 1.4, "temp": 23.5, "status": "High Purity Riverine Aquifer Water", "pot": "Compliant (Safe for Drinking)"},
    # International (USGS)
    "342500118250001": {"ph": 8.1, "cond": 1250, "tds": 780, "nitrate": 11.2, "temp": 20.8, "status": "High Salinity & Nitrate (Exceeds EPA 10 mg/L)", "pot": "Non-Compliant / Warning"},
    "293830098255501": {"ph": 7.4, "cond": 480, "tds": 310, "nitrate": 2.5, "temp": 22.1, "status": "Good / Pristine Karst Water", "pot": "Compliant"},
    "280255082260501": {"ph": 7.8, "cond": 390, "tds": 240, "nitrate": 0.9, "temp": 23.4, "status": "High Purity / Low Mineralization", "pot": "Compliant"},
    "332450111584001": {"ph": 8.2, "cond": 1420, "tds": 890, "nitrate": 4.1, "temp": 25.6, "status": "High Dissolved Solids / Mineralized", "pot": "High Mineralization"},
    "405012100123401": {"ph": 7.6, "cond": 620, "tds": 410, "nitrate": 6.8, "temp": 15.2, "status": "Moderate Nitrate / Agricultural Influence", "pot": "Compliant"},
    "01646500": {"ph": 7.3, "cond": 340, "tds": 210, "nitrate": 1.8, "temp": 14.5, "status": "Good / Potable", "pot": "Compliant"}
}

quality_rows = []
for site in sites_data:
    sno = site["site_no"]
    prof = quality_profiles.get(sno, {"ph": 7.4, "cond": 550, "tds": 350, "nitrate": 4.5, "temp": 24.0, "status": "Potable", "pot": "Compliant"})
    quality_rows.append({
        "site_no": sno,
        "station_nm": site["station_nm"],
        "sample_date": "2024-10-15",
        "ph": prof["ph"],
        "specific_conductance_us_cm": prof["cond"],
        "total_dissolved_solids_mg_l": prof["tds"],
        "nitrate_mg_l": prof["nitrate"],
        "water_temperature_c": prof["temp"],
        "water_quality_assessment": prof["status"],
        "epa_potability": prof["pot"]
    })

df_quality = pd.DataFrame(quality_rows)
quality_csv_path = os.path.join(DATA_DIR, "usgs_water_quality.csv")
df_quality.to_csv(quality_csv_path, index=False)
print(f"Saved {len(df_quality)} water quality records to {quality_csv_path}")

# 4. Rich Training Queries for Intent Classification (India + Global)
training_queries = [
    # INDIA SPECIFIC QUERIES
    ("What is the groundwater level in Punjab Ludhiana?", "depth", "CGWB-PB-LUD01"),
    ("How deep is the water table in Amritsar right now?", "depth", "CGWB-PB-AMR02"),
    ("Check groundwater depth in Karnal Haryana", "depth", "CGWB-HR-KAR01"),
    ("Current water table depth in Gurugram Gurgaon", "depth", "CGWB-HR-GUR02"),
    ("How far down is groundwater in New Delhi NCR?", "depth", "CGWB-DL-DEL01"),
    ("What is the water level in Jodhpur Rajasthan desert?", "depth", "CGWB-RJ-JOD01"),
    ("Check water depth in Jaipur Rajasthan", "depth", "CGWB-RJ-JAI02"),
    ("How deep is the well in Jaisalmer Thar desert?", "depth", "CGWB-RJ-JAI03"),
    ("Water level in Varanasi Uttar Pradesh Gangetic plain", "depth", "CGWB-UP-VAR01"),
    ("Current groundwater depth in Lucknow UP", "depth", "CGWB-UP-LUC02"),
    ("How deep is water table in Agra Uttar Pradesh?", "depth", "CGWB-UP-AGR03"),
    ("What is the water depth in Pune Maharashtra Deccan plateau?", "depth", "CGWB-MH-PUN01"),
    ("Check groundwater depletion in Solapur Marathwada", "depth", "CGWB-MH-SOL02"),
    ("Current water level in Nagpur Vidarbha Maharashtra", "depth", "CGWB-MH-NAG03"),
    ("How deep is groundwater table in Ahmedabad Gujarat?", "depth", "CGWB-GJ-AHM01"),
    ("Water table depth in Rajkot Saurashtra Gujarat", "depth", "CGWB-GJ-RAJ02"),
    ("What is the groundwater depth in Bengaluru Bangalore Karnataka?", "depth", "CGWB-KA-BLR01"),
    ("Check water level in Mysuru Mysore Karnataka", "depth", "CGWB-KA-MYS02"),
    ("How deep is groundwater in Chennai Tamil Nadu coastal well?", "depth", "CGWB-TN-CHE01"),
    ("Water table measurement in Coimbatore Tamil Nadu", "depth", "CGWB-TN-COI02"),
    ("Current water level in Hyderabad Telangana", "depth", "CGWB-TS-HYD01"),
    ("What is the groundwater depth in Kolkata West Bengal?", "depth", "CGWB-WB-KOL01"),
    ("How far down to hit groundwater in Patna Bihar?", "depth", "CGWB-BR-PAT01"),
    ("What is the water level in Kochi Ernakulam Kerala?", "depth", "CGWB-KL-ERN01"),
    ("Groundwater level in Indore Madhya Pradesh", "depth", "CGWB-MP-IND01"),
    ("Check water depth in Guwahati Assam Brahmaputra valley", "depth", "CGWB-AS-GUW01"),

    # SOIL MOISTURE & COMPREHENSIVE DOSSIER QUERIES
    ("What is the soil moisture and water table in Punjab?", "location", "CGWB-PB-LUD01"),
    ("Tell me about groundwater and soil moisture in Rajasthan", "location", "CGWB-RJ-JOD01"),
    ("Soil moisture and drought condition in Jodhpur", "location", "CGWB-RJ-JOD01"),
    ("Give complete groundwater and soil moisture report for Maharashtra Pune", "location", "CGWB-MH-PUN01"),
    ("Check soil condition and water level in Solapur", "location", "CGWB-MH-SOL02"),
    ("What is the soil moisture in Kerala Ernakulam?", "location", "CGWB-KL-ERN01"),
    ("Is soil dry or moist in Gujarat Ahmedabad?", "location", "CGWB-GJ-AHM01"),
    ("Check soil moisture in Uttar Pradesh Varanasi", "location", "CGWB-UP-VAR01"),
    ("Soil moisture and groundwater level in Bengaluru Karnataka", "location", "CGWB-KA-BLR01"),
    ("Everything about groundwater and soil in Kolkata West Bengal", "location", "CGWB-WB-KOL01"),
    ("Full hydrological dossier for Delhi NCR", "location", "CGWB-DL-DEL01"),
    ("Check soil moisture and aquifer in Kamrup Assam", "location", "CGWB-AS-GUW01"),
    ("Soil moisture and agricultural water in Patna Bihar", "location", "CGWB-BR-PAT01"),
    ("What is the soil condition in Indore Madhya Pradesh?", "location", "CGWB-MP-IND01"),
    ("Tell me about groundwater and soil moisture in California", "location", "342500118250001"),
    ("What is the groundwater level and soil condition in Kern County?", "location", "342500118250001"),
    ("Give me complete groundwater and soil moisture report for Texas Bexar", "location", "293830098255501"),

    # TREND & HYDROGRAPH QUERIES
    ("Show groundwater level trend over time in Punjab Ludhiana", "trend", "CGWB-PB-LUD01"),
    ("Is groundwater declining or rising in Amritsar?", "trend", "CGWB-PB-AMR02"),
    ("Show historical 24 month water fluctuation in Jodhpur Rajasthan", "trend", "CGWB-RJ-JOD01"),
    ("Plot historical time series hydrograph for Delhi NCR", "trend", "CGWB-DL-DEL01"),
    ("Monsoon recharge and water level trend in Pune Maharashtra", "trend", "CGWB-MH-PUN01"),
    ("Analyze water depletion history in Solapur drought zone", "trend", "CGWB-MH-SOL02"),
    ("Show water level changes over past 2 years in Varanasi UP", "trend", "CGWB-UP-VAR01"),
    ("Historical hydrograph for Bengaluru Bangalore Karnataka", "trend", "CGWB-KA-BLR01"),
    ("Long term groundwater trend in Chennai Tamil Nadu", "trend", "CGWB-TN-CHE01"),
    ("Water level trend curve in Kolkata West Bengal", "trend", "CGWB-WB-KOL01"),
    ("Has the water table changed historically in Ahmedabad Gujarat?", "trend", "CGWB-GJ-AHM01"),
    ("Seasonal fluctuation in Kochi Kerala", "trend", "CGWB-KL-ERN01"),
    ("Show groundwater level trend over time for site 01646500", "trend", "01646500"),
    ("Is the water table declining or rising in well 342500118250001?", "trend", "342500118250001"),
    ("Historical water level fluctuations for Edwards Aquifer well 293830098255501", "trend", "293830098255501"),

    # QUALITY & POTABILITY QUERIES
    ("Is groundwater safe for drinking in Ludhiana Punjab?", "quality", "CGWB-PB-LUD01"),
    ("Check pH, nitrate and TDS for Delhi groundwater", "quality", "CGWB-DL-DEL01"),
    ("What is the water quality in Jodhpur Rajasthan?", "quality", "CGWB-RJ-JOD01"),
    ("Is there high salinity in Jaisalmer water?", "quality", "CGWB-RJ-JAI03"),
    ("Check drinking water potability in Varanasi Uttar Pradesh", "quality", "CGWB-UP-VAR01"),
    ("Water chemical test results for Pune Maharashtra", "quality", "CGWB-MH-PUN01"),
    ("Is water safe for consumption in Bengaluru Karnataka?", "quality", "CGWB-KA-BLR01"),
    ("Check coastal salinity and TDS in Chennai Tamil Nadu", "quality", "CGWB-TN-CHE01"),
    ("What is the water quality and potability in Kolkata?", "quality", "CGWB-WB-KOL01"),
    ("Is water pristine in Kochi Kerala?", "quality", "CGWB-KL-ERN01"),
    ("Check TDS and nitrate for Ahmedabad Gujarat", "quality", "CGWB-GJ-AHM01"),
    ("What is the water quality at site 01646500?", "quality", "01646500"),
    ("Is the groundwater safe for drinking in Kern County well 342500118250001?", "quality", "342500118250001"),
    ("Show pH, nitrate, and conductivity measurements for well 293830098255501", "quality", "293830098255501"),
    ("Check total dissolved solids and nitrate concentration for 405012100123401", "quality", "405012100123401"),

    # LOCATION & METADATA QUERIES
    ("Where is monitoring station CGWB-PB-LUD01 located?", "location", "CGWB-PB-LUD01"),
    ("Find GPS coordinates of station in Jodhpur Rajasthan", "location", "CGWB-RJ-JOD01"),
    ("Which aquifer is tapped in Ludhiana Punjab?", "location", "CGWB-PB-LUD01"),
    ("Locate the monitoring station in Pune Maharashtra", "location", "CGWB-MH-PUN01"),
    ("What aquifer does Bengaluru well draw from?", "location", "CGWB-KA-BLR01"),
    ("Show latitude and longitude for Varanasi station", "location", "CGWB-UP-VAR01"),
    ("Where is groundwater monitoring station 01646500 located?", "location", "01646500"),
    ("Find the GPS coordinates and latitude longitude for well 342500118250001", "location", "342500118250001"),
    ("Which county and state does site 293830098255501 belong to?", "location", "293830098255501"),
    ("Where can I find the Floridan Aquifer monitoring well?", "location", "280255082260501")
]

df_train = pd.DataFrame(training_queries, columns=["query_text", "intent", "target_site_no"])
train_csv_path = os.path.join(DATA_DIR, "training_queries.csv")
df_train.to_csv(train_csv_path, index=False)
print(f"Saved {len(df_train)} training queries to {train_csv_path}")

# 5. Benchmark Test Suite
benchmark_tests = [
    {
        "test_group": "Group 1: Indian Alluvial Lookups (Punjab / UP)",
        "category": "depth",
        "target_site_no": "CGWB-PB-LUD01",
        "phrasings": [
            "What is the groundwater level in Punjab Ludhiana?",
            "How deep down is the water table in Ludhiana right now?",
            "Water table depth measurement for Ludhiana CGWB well",
            "Give me recent groundwater reading in Ludhiana Punjab",
            "How deep is the tube well water in Ludhiana?"
        ]
    },
    {
        "test_group": "Group 2: Arid Rajasthan Groundwater & Soil",
        "category": "location",
        "target_site_no": "CGWB-RJ-JOD01",
        "phrasings": [
            "Tell me about groundwater and soil moisture in Rajasthan",
            "What is the soil moisture and water table in Jodhpur?",
            "Comprehensive groundwater and soil report for Jodhpur Rajasthan",
            "Check soil dryness and groundwater in Jodhpur",
            "Where is Jodhpur monitoring station and how is the soil?"
        ]
    },
    {
        "test_group": "Group 3: Maharashtra Basalt Trend & Fluctuation",
        "category": "trend",
        "target_site_no": "CGWB-MH-PUN01",
        "phrasings": [
            "Show groundwater level trend over time in Pune Maharashtra",
            "Monsoon recharge and water level trend in Pune Maharashtra",
            "Has the water table been declining or rising in Pune?",
            "24 month water level fluctuations in Pune Deccan basalt",
            "Hydrograph of water table changes for Pune station"
        ]
    },
    {
        "test_group": "Group 4: Potability & Quality Check in Gangetic Plain",
        "category": "quality",
        "target_site_no": "CGWB-UP-VAR01",
        "phrasings": [
            "Check drinking water potability in Varanasi Uttar Pradesh",
            "Is groundwater safe for drinking in Varanasi UP?",
            "What are the pH, TDS, and nitrate levels in Varanasi water?",
            "Check chemical potability and contamination status for Varanasi",
            "Does Varanasi groundwater meet BIS drinking water standards?"
        ]
    },
    {
        "test_group": "Group 5: US Central Valley Depth Lookups",
        "category": "depth",
        "target_site_no": "342500118250001",
        "phrasings": [
            "What is the current water level at well 342500118250001?",
            "How deep down is the water in well 342500118250001 right now?",
            "Latest water table depth below surface for 342500118250001",
            "Give me the most recent water level reading for station 342500118250001",
            "How far down do we have to drill to reach groundwater at 342500118250001?"
        ]
    },
    {
        "test_group": "Group 6: Historical Trend Disambiguation",
        "category": "trend",
        "target_site_no": "293830098255501",
        "phrasings": [
            "Show groundwater level trend over time for site 293830098255501",
            "Has the water table been declining or rising in well 293830098255501?",
            "Historical fluctuation chart of water level at Edwards Aquifer 293830098255501",
            "Show me seasonal drawdown time-series for station 293830098255501",
            "Give me multi-month groundwater depletion curve for site 293830098255501"
        ]
    }
]

test_json_path = os.path.join(DATA_DIR, "benchmark_test_queries.json")
with open(test_json_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_tests, f, indent=2)
print(f"Saved {len(benchmark_tests)} benchmark test groups to {test_json_path}")
