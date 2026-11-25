import os
import pandas as pd
from typing import Dict, Any, Optional, List

class USGSDataLoader:
    """
    Loads and provides structured querying capabilities over India CGWB & Global USGS NWIS datasets,
    including groundwater levels, historical trends, water quality, and USDA/NASA/ISRO soil moisture.
    """
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
        self.data_dir = data_dir
        
        self.sites_df = pd.read_csv(os.path.join(data_dir, "usgs_groundwater_sites.csv"), dtype={"site_no": str})
        self.timeseries_df = pd.read_csv(os.path.join(data_dir, "usgs_water_levels_timeseries.csv"), dtype={"site_no": str})
        self.quality_df = pd.read_csv(os.path.join(data_dir, "usgs_water_quality.csv"), dtype={"site_no": str})
        
        self.sites_df["site_no"] = self.sites_df["site_no"].astype(str)
        self.timeseries_df["site_no"] = self.timeseries_df["site_no"].astype(str)
        self.quality_df["site_no"] = self.quality_df["site_no"].astype(str)
        
        # Build lookup indices
        self.sites_by_id = {row["site_no"]: row.to_dict() for _, row in self.sites_df.iterrows()}
        self.counties = {str(c).lower(): c for c in self.sites_df["county"].dropna().unique()}
        self.states = {str(s).lower(): s for s in self.sites_df["state"].dropna().unique()}
        self.state_codes = {str(sc).lower(): sc for sc in self.sites_df["state_code"].dropna().unique()}
        self.aquifers = {str(a).lower(): a for a in self.sites_df["aquifer"].dropna().unique()}

    def get_all_sites(self) -> pd.DataFrame:
        return self.sites_df

    def get_site(self, site_no: str) -> Optional[Dict[str, Any]]:
        return self.sites_by_id.get(site_no)

    def find_matching_sites(self, entities: Dict[str, Any], query: str = "") -> List[Dict[str, Any]]:
        """
        Returns list of candidate sites matching entities or place query.
        """
        results = []
        if entities.get("site_no"):
            s = self.get_site(entities["site_no"])
            if s:
                results.append(s)
        if query:
            p = self.search_by_place_name(query)
            if p and p not in results:
                results.append(p)
        if not results:
            e = self.find_site_by_entities(entities)
            if e and e not in results:
                results.append(e)
        return results

    def find_site_by_entities(self, entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolves the station from extracted entities (site_no, county/district, state, country, aquifer).
        """
        if "site_no" in entities and entities["site_no"]:
            s = self.get_site(entities["site_no"])
            if s:
                return s

        matches = self.sites_df.copy()
        if "county" in entities and entities["county"]:
            matches = matches[matches["county"].str.lower() == str(entities["county"]).lower()]
            if not matches.empty:
                return matches.iloc[0].to_dict()

        if "state" in entities and entities["state"]:
            matches = self.sites_df[(self.sites_df["state"].str.lower() == str(entities["state"]).lower()) | 
                                   (self.sites_df["state_code"].str.lower() == str(entities["state"]).lower())]
            if not matches.empty:
                return matches.iloc[0].to_dict()

        if "aquifer" in entities and entities["aquifer"]:
            matches = self.sites_df[self.sites_df["aquifer"].str.lower().str.contains(str(entities["aquifer"]).lower())]
            if not matches.empty:
                return matches.iloc[0].to_dict()
        
        if "country" in entities and entities["country"]:
            matches = self.sites_df[self.sites_df["country"].str.lower() == str(entities["country"]).lower()]
            if not matches.empty:
                return matches.iloc[0].to_dict()

        return None

    def search_by_place_name(self, place_query: str, allow_default: bool = True) -> Optional[Dict[str, Any]]:
        """
        Searches for a site matching any free-form place name (Indian district, state, or international).
        """
        if not place_query or not isinstance(place_query, str):
            return self.sites_df.iloc[0].to_dict() if allow_default else None
            
        q_lower = place_query.lower()
        
        # 1. Direct site number search
        for sno, site in self.sites_by_id.items():
            if sno.lower() in q_lower:
                return site
                
        # 2. District / County search
        for c_lower, c_proper in self.counties.items():
            if c_lower in q_lower:
                subset = self.sites_df[self.sites_df["county"].str.lower() == c_lower]
                if not subset.empty:
                    return subset.iloc[0].to_dict()
                    
        # 3. State search
        for s_lower, s_proper in self.states.items():
            if s_lower in q_lower:
                subset = self.sites_df[self.sites_df["state"].str.lower() == s_lower]
                if not subset.empty:
                    return subset.iloc[0].to_dict()
                    
        # 4. State code search
        for sc_lower, sc_proper in self.state_codes.items():
            if f" {sc_lower} " in f" {q_lower} " or q_lower.endswith(f" {sc_lower}"):
                subset = self.sites_df[self.sites_df["state_code"].str.lower() == sc_lower]
                if not subset.empty:
                    return subset.iloc[0].to_dict()
                    
        # 5. Aquifer keyword search
        for a_lower, a_proper in self.aquifers.items():
            if any(term in q_lower for term in a_lower.split() if len(term) > 4):
                subset = self.sites_df[self.sites_df["aquifer"].str.lower() == a_lower]
                if not subset.empty:
                    return subset.iloc[0].to_dict()

        # 6. Station name substring
        for _, row in self.sites_df.iterrows():
            station_parts = [p.lower() for p in str(row["station_nm"]).split() if len(p) > 3]
            if any(part in q_lower for part in station_parts):
                return row.to_dict()

        # If user mentioned India, default to a major Indian station (Ludhiana or Delhi)
        if "india" in q_lower:
            subset = self.sites_df[self.sites_df["country"] == "India"]
            if not subset.empty:
                return subset.iloc[0].to_dict()

        return self.sites_df.iloc[0].to_dict() if allow_default else None

    def get_timeseries(self, site_no: str) -> pd.DataFrame:
        df = self.timeseries_df[self.timeseries_df["site_no"] == site_no].copy()
        df["record_date"] = pd.to_datetime(df["record_date"])
        df = df.sort_values("record_date")
        return df

    def get_quality(self, site_no: str) -> Optional[Dict[str, Any]]:
        subset = self.quality_df[self.quality_df["site_no"] == site_no]
        if not subset.empty:
            return subset.iloc[0].to_dict()
        return None

    def get_full_location_dossier(self, site_no: str) -> Dict[str, Any]:
        """
        Returns a comprehensive environmental & hydrological dossier:
        Groundwater Depth (in both meters & feet), Soil Moisture, Historical Trend, Water Quality, and Aquifer details.
        """
        site = self.get_site(site_no)
        if not site:
            site = self.sites_df.iloc[0].to_dict()
            site_no = site["site_no"]

        ts = self.get_timeseries(site_no)
        qual = self.get_quality(site_no)
        
        # Calculate trend dynamics
        net_change = 0.0
        trend_direction = "Stable"
        min_depth = site["current_water_level_ft"]
        max_depth = site["current_water_level_ft"]
        if not ts.empty:
            first_depth = ts.iloc[0]["water_level_depth_ft"]
            last_depth = ts.iloc[-1]["water_level_depth_ft"]
            net_change = round(last_depth - first_depth, 2)
            min_depth = round(ts["water_level_depth_ft"].min(), 2)
            max_depth = round(ts["water_level_depth_ft"].max(), 2)
            trend_direction = "Declining (Water Table Dropping)" if net_change > 0 else "Rising (Active Recharge)" if net_change < 0 else "Stable"

        depth_ft = site["current_water_level_ft"]
        depth_m = round(depth_ft * 0.3048, 2)

        return {
            "site_no": site_no,
            "station_nm": site["station_nm"],
            "country": site.get("country", "India" if "CGWB" in site_no else "United States"),
            "state": site["state"],
            "county": site["county"],
            "latitude": site["dec_lat_va"],
            "longitude": site["dec_long_va"],
            "aquifer": site["aquifer"],
            "altitude_ft": site["altitude_va"],
            "well_depth_ft": site["well_depth_va"],
            
            # 1. Groundwater Depth (Feet and Meters)
            "current_water_level_ft": depth_ft,
            "current_water_level_m": depth_m,
            "water_level_elevation_ft": round(site["altitude_va"] - depth_ft, 2),
            "water_scarcity_index": site.get("water_scarcity_index", "Active Monitoring"),
            
            # 2. Soil Moisture (Root Zone and Surface)
            "surface_soil_moisture_pct": site.get("surface_soil_moisture_pct", 22.0),
            "rootzone_soil_moisture_pct": site.get("rootzone_soil_moisture_pct", 26.0),
            "soil_moisture_condition": site.get("soil_moisture_condition", "Moderate"),
            "soil_type": site.get("soil_type", "Loam"),
            
            # 3. 24-Month Trend
            "timeseries_data": ts,
            "net_change_ft": net_change,
            "net_change_m": round(net_change * 0.3048, 2),
            "trend_direction": trend_direction,
            "min_depth_ft": min_depth,
            "max_depth_ft": max_depth,
            "history_months": len(ts),
            
            # 4. Water Quality
            "ph": qual["ph"] if qual else 7.2,
            "nitrate_mg_l": qual["nitrate_mg_l"] if qual else 4.0,
            "total_dissolved_solids_mg_l": qual["total_dissolved_solids_mg_l"] if qual else 420,
            "specific_conductance_us_cm": qual["specific_conductance_us_cm"] if qual else 600,
            "epa_potability": qual["epa_potability"] if qual else "Compliant",
            "water_quality_assessment": qual["water_quality_assessment"] if qual else "Standard"
        }

    def build_decision_summary(self, site: Dict[str, Any], intent: str, record: Dict[str, Any], dossier: Dict[str, Any]) -> str:
        """
        Produces field-decision takeaway closing the decision loop.
        """
        sname = site.get("station_nm", "Monitoring Station")
        loc = f"{site.get('county', '')}, {site.get('state', '')}"
        depth_ft = dossier.get("current_water_level_ft", 0)
        depth_m = dossier.get("current_water_level_m", round(depth_ft * 0.3048, 1))
        soil = dossier.get("soil_moisture_condition", "Moderate")
        pot = dossier.get("epa_potability", "Compliant")
        trend = dossier.get("trend_direction", "Stable")
        
        if intent == "depth":
            return f"Water table at {sname} ({loc}) is {depth_ft} ft ({depth_m} m) below surface. Scarcity: {dossier.get('water_scarcity_index', 'Monitored')}."
        elif intent == "trend":
            return f"24-month trend at {sname} ({loc}) is {trend} with net change {dossier.get('net_change_ft', 0):+.2f} ft ({dossier.get('net_change_m', 0):+.2f} m)."
        elif intent == "quality":
            return f"Water quality at {sname} ({loc}) is {pot} (pH: {dossier.get('ph')}, Nitrate: {dossier.get('nitrate_mg_l')} mg/L, TDS: {dossier.get('total_dissolved_solids_mg_l')} mg/L)."
        else:
            return f"{sname} is in {loc} tapping {site.get('aquifer', 'Aquifer')}. Current depth: {depth_ft} ft ({depth_m} m), Soil: {soil}."

    def get_structured_record(self, site_no: str, intent: str) -> Dict[str, Any]:
        """
        Returns structured payload matching the identified retrieval category.
        """
        site = self.get_site(site_no)
        if not site:
            site = self.sites_df.iloc[0].to_dict()
            site_no = site["site_no"]
            
        depth_ft = site["current_water_level_ft"]
        depth_m = round(depth_ft * 0.3048, 2)
        
        result = {
            "site_no": site_no,
            "station_nm": site["station_nm"],
            "country": site.get("country", "India"),
            "state": site["state"],
            "county": site["county"],
            "aquifer": site["aquifer"],
            "intent": intent,
            "surface_soil_moisture_pct": site.get("surface_soil_moisture_pct", 22.0),
            "rootzone_soil_moisture_pct": site.get("rootzone_soil_moisture_pct", 26.0),
            "soil_moisture_condition": site.get("soil_moisture_condition", "Moderate"),
            "soil_type": site.get("soil_type", "Loam"),
            "water_scarcity_index": site.get("water_scarcity_index", "Active Monitoring"),
            "current_water_level_ft": depth_ft,
            "current_water_level_m": depth_m
        }
        
        if intent == "location":
            result.update({
                "latitude": site["dec_lat_va"],
                "longitude": site["dec_long_va"],
                "altitude_ft": site["altitude_va"],
                "well_depth_ft": site["well_depth_va"],
                "status": site["status"],
                "water_level_elevation_ft": round(site["altitude_va"] - depth_ft, 2)
            })
        elif intent == "depth":
            result.update({
                "current_water_level_ft": depth_ft,
                "current_water_level_m": depth_m,
                "water_level_elevation_ft": round(site["altitude_va"] - depth_ft, 2),
                "altitude_ft": site["altitude_va"],
                "measurement_status": "Approved Active Measurement"
            })
        elif intent == "trend":
            ts = self.get_timeseries(site_no)
            if not ts.empty:
                first_depth = ts.iloc[0]["water_level_depth_ft"]
                last_depth = ts.iloc[-1]["water_level_depth_ft"]
                net_change = round(last_depth - first_depth, 2)
                trend_direction = "Declining (Water Table Dropping)" if net_change > 0 else "Rising (Recharge)" if net_change < 0 else "Stable"
                result.update({
                    "timeseries_data": ts,
                    "record_count": len(ts),
                    "start_date": ts.iloc[0]["record_date"].strftime("%Y-%m-%d"),
                    "end_date": ts.iloc[-1]["record_date"].strftime("%Y-%m-%d"),
                    "net_change_ft": net_change,
                    "trend_direction": trend_direction
                })
        elif intent == "quality":
            q = self.get_quality(site_no)
            if q:
                result.update({
                    "ph": q["ph"],
                    "specific_conductance_us_cm": q["specific_conductance_us_cm"],
                    "total_dissolved_solids_mg_l": q["total_dissolved_solids_mg_l"],
                    "nitrate_mg_l": q["nitrate_mg_l"],
                    "water_temperature_c": q["water_temperature_c"],
                    "water_quality_assessment": q["water_quality_assessment"],
                    "epa_potability": q["epa_potability"]
                })
        return result
