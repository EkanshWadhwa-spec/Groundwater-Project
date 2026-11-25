import re
import string
from typing import Dict, Any, List, Tuple

# Stopwords for query normalization
STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", 
    "about", "by", "of", "from", "is", "are", "was", "were", "be", "been", 
    "being", "have", "has", "had", "do", "does", "did", "can", "could", 
    "will", "would", "shall", "should", "tell", "show", "give", "me", "get", 
    "check", "find", "fetch", "look", "up", "display", "view", "i", "we", "you", "please",
    "everything", "something", "related", "like", "na", "yrr", "yrrr", "bro"
}

# Indian States & International States
KNOWN_STATES = {
    # India
    "punjab": "PB", "haryana": "HR", "delhi": "DL", "rajasthan": "RJ",
    "uttar pradesh": "UP", "maharashtra": "MH", "gujarat": "GJ",
    "karnataka": "KA", "tamil nadu": "TN", "telangana": "TS",
    "west bengal": "WB", "bihar": "BR", "kerala": "KL",
    "madhya pradesh": "MP", "assam": "AS", "andhra pradesh": "AP",
    "odisha": "OD", "himachal pradesh": "HP", "uttarakhand": "UK",
    # US
    "california": "CA", "texas": "TX", "florida": "FL",
    "arizona": "AZ", "nebraska": "NE", "maryland": "MD"
}

KNOWN_STATE_CODES = {
    "pb", "hr", "dl", "rj", "up", "mh", "gj", "ka", "tn", "ts", "wb", "br", "kl", "mp", "as",
    "ca", "tx", "fl", "az", "ne", "md"
}

# Indian Districts / Cities & US Counties
KNOWN_DISTRICTS_AND_CITIES = {
    # Punjab
    "ludhiana": ("Punjab", "Ludhiana"),
    "amritsar": ("Punjab", "Amritsar"),
    "jalandhar": ("Punjab", "Jalandhar"),
    "patiala": ("Punjab", "Patiala"),
    
    # Haryana
    "karnal": ("Haryana", "Karnal"),
    "gurugram": ("Haryana", "Gurugram"),
    "gurgaon": ("Haryana", "Gurugram"),
    "faridabad": ("Haryana", "Faridabad"),
    "kurukshetra": ("Haryana", "Karnal"),
    
    # Delhi
    "new delhi": ("Delhi", "New Delhi"),
    "delhi": ("Delhi", "New Delhi"),
    "ncr": ("Delhi", "New Delhi"),
    
    # Rajasthan
    "jodhpur": ("Rajasthan", "Jodhpur"),
    "jaipur": ("Rajasthan", "Jaipur"),
    "jaisalmer": ("Rajasthan", "Jaisalmer"),
    "bikaner": ("Rajasthan", "Jodhpur"),
    "udaipur": ("Rajasthan", "Jaipur"),
    
    # Uttar Pradesh
    "varanasi": ("Uttar Pradesh", "Varanasi"),
    "banaras": ("Uttar Pradesh", "Varanasi"),
    "kashi": ("Uttar Pradesh", "Varanasi"),
    "lucknow": ("Uttar Pradesh", "Lucknow"),
    "kanpur": ("Uttar Pradesh", "Lucknow"),
    "agra": ("Uttar Pradesh", "Agra"),
    "prayagraj": ("Uttar Pradesh", "Varanasi"),
    "allahabad": ("Uttar Pradesh", "Varanasi"),
    
    # Maharashtra
    "pune": ("Maharashtra", "Pune"),
    "mumbai": ("Maharashtra", "Pune"),
    "solapur": ("Maharashtra", "Solapur"),
    "nagpur": ("Maharashtra", "Nagpur"),
    "vidarbha": ("Maharashtra", "Nagpur"),
    "marathwada": ("Maharashtra", "Solapur"),
    "aurangabad": ("Maharashtra", "Pune"),
    "sambhaji nagar": ("Maharashtra", "Pune"),
    "nashik": ("Maharashtra", "Pune"),
    
    # Gujarat
    "ahmedabad": ("Gujarat", "Ahmedabad"),
    "rajkot": ("Gujarat", "Rajkot"),
    "saurashtra": ("Gujarat", "Rajkot"),
    "surat": ("Gujarat", "Ahmedabad"),
    "vadodara": ("Gujarat", "Ahmedabad"),
    
    # Karnataka
    "bengaluru": ("Karnataka", "Bengaluru"),
    "bangalore": ("Karnataka", "Bengaluru"),
    "mysuru": ("Karnataka", "Mysuru"),
    "mysore": ("Karnataka", "Mysuru"),
    
    # Tamil Nadu
    "chennai": ("Tamil Nadu", "Chennai"),
    "madras": ("Tamil Nadu", "Chennai"),
    "coimbatore": ("Tamil Nadu", "Coimbatore"),
    "madurai": ("Tamil Nadu", "Chennai"),
    
    # Telangana
    "hyderabad": ("Telangana", "Hyderabad"),
    "secunderabad": ("Telangana", "Hyderabad"),
    
    # West Bengal
    "kolkata": ("West Bengal", "Kolkata"),
    "calcutta": ("West Bengal", "Kolkata"),
    "howrah": ("West Bengal", "Kolkata"),
    
    # Bihar
    "patna": ("Bihar", "Patna"),
    "gaya": ("Bihar", "Patna"),
    
    # Kerala
    "ernakulam": ("Kerala", "Ernakulam"),
    "kochi": ("Kerala", "Ernakulam"),
    "cochin": ("Kerala", "Ernakulam"),
    "thiruvananthapuram": ("Kerala", "Ernakulam"),
    "trivandrum": ("Kerala", "Ernakulam"),
    
    # Madhya Pradesh
    "indore": ("Madhya Pradesh", "Indore"),
    "bhopal": ("Madhya Pradesh", "Indore"),
    "ujjain": ("Madhya Pradesh", "Indore"),
    
    # Assam
    "guwahati": ("Assam", "Kamrup"),
    "kamrup": ("Assam", "Kamrup"),
    
    # US
    "kern": ("California", "Kern"),
    "fresno": ("California", "Fresno"),
    "bexar": ("Texas", "Bexar"),
    "medina": ("Texas", "Medina"),
    "buffalo": ("Nebraska", "Buffalo"),
    "marion": ("Florida", "Marion"),
    "orange": ("Florida", "Orange"),
    "maricopa": ("Arizona", "Maricopa"),
    "phoenix": ("Arizona", "Maricopa"),
    "montgomery": ("Maryland", "Montgomery")
}

KNOWN_AQUIFERS = {
    "indo-gangetic": "Indo-Gangetic Alluvial Aquifer System",
    "alluvial": "Indo-Gangetic Alluvial Aquifer System",
    "gangetic": "Central Gangetic Plain Alluvium",
    "deccan": "Deccan Traps Basaltic Hard Rock Aquifer",
    "basalt": "Deccan Traps Basaltic Hard Rock Aquifer",
    "marwar": "Jodhpur Sandstone & Fractured Rock Aquifer",
    "lathi": "Lathi Sandstone Deep Aquifer",
    "peninsular": "Peninsular Gneiss & Granitic Hard Rock Aquifer",
    "granite": "Peninsular Gneiss & Granitic Hard Rock Aquifer",
    "brahmaputra": "Brahmaputra Valley Alluvial Aquifer",
    "edwards": "Edwards-Trinity Aquifer System",
    "ogallala": "High Plains (Ogallala) Aquifer",
    "floridan": "Floridan Aquifer System",
    "central valley": "California Central Valley Aquifer System"
}

class QueryPreprocessor:
    """
    Normalizes natural-language groundwater queries and extracts structured entities
    for Indian states & districts as well as international locations.
    """
    def __init__(self):
        pass

    def normalize(self, query: str) -> str:
        if not query or not isinstance(query, str):
            return ""
        
        text = query.lower().strip()
        text = re.sub(r'(\b[a-z])-([0-9]+\b)', r'\1\2', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        filtered = [tok for tok in tokens if tok not in STOPWORDS]
        return " ".join(filtered)

    def normalize_for_intent(self, query: str) -> str:
        """
        Entity-disentangled normalization for TF-IDF intent classification.
        Strips specific site numbers and place names so intent classification
        is purely driven by intent-bearing vocabulary (level, depth, trend, quality, potability, soil, moisture).
        """
        if not query or not isinstance(query, str):
            return ""
        text = query.lower().strip()
        # strip site numbers
        text = re.sub(r'\b(cgwb-[a-z]{2}-[a-z0-9]{3,6}|0[0-9]{7}|[1-9][0-9]{7,14})\b', ' ', text)
        # strip digits
        text = re.sub(r'\b\d+\b', ' ', text)
        # strip known districts / cities
        for d in KNOWN_DISTRICTS_AND_CITIES:
            text = re.sub(r'\b' + re.escape(d) + r'\b', ' ', text)
        # strip known states
        for s in KNOWN_STATES:
            text = re.sub(r'\b' + re.escape(s) + r'\b', ' ', text)
        for sc in KNOWN_STATE_CODES:
            text = re.sub(r'\b' + re.escape(sc) + r'\b', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        filtered = [tok for tok in tokens if tok not in STOPWORDS]
        if not filtered:
            return self.normalize(query)
        return " ".join(filtered)

    def extract_entities(self, query: str) -> Dict[str, Any]:
        raw_text = query.lower()
        entities = {
            "site_no": None,
            "country": None,
            "state": None,
            "county": None,
            "aquifer": None,
            "place_name": None
        }
        
        # 1. Site Number match (CGWB or USGS format)
        cgwb_match = re.search(r'\b(cgwb-[a-z]{2}-[a-z0-9]{3,6})\b', raw_text)
        if cgwb_match:
            entities["site_no"] = cgwb_match.group(1).upper()
            entities["country"] = "India"
        else:
            usgs_match = re.search(r'\b(0[0-9]{7}|[1-9][0-9]{7,14})\b', raw_text)
            if usgs_match:
                entities["site_no"] = usgs_match.group(1)

        # 2. Match Districts & Cities
        for place, (p_state, p_county) in KNOWN_DISTRICTS_AND_CITIES.items():
            if re.search(r'\b' + re.escape(place) + r'\b', raw_text):
                entities["state"] = p_state
                entities["county"] = p_county
                entities["place_name"] = place.title()
                if p_state in ["Punjab", "Haryana", "Delhi", "Rajasthan", "Uttar Pradesh", "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Telangana", "West Bengal", "Bihar", "Kerala", "Madhya Pradesh", "Assam"]:
                    entities["country"] = "India"
                else:
                    entities["country"] = "United States"
                break

        # 3. Match State if not found
        if not entities["state"]:
            for state_name, code in KNOWN_STATES.items():
                if re.search(r'\b' + re.escape(state_name) + r'\b', raw_text):
                    entities["state"] = state_name.title()
                    entities["place_name"] = state_name.title()
                    if state_name in ["punjab", "haryana", "delhi", "rajasthan", "uttar pradesh", "maharashtra", "gujarat", "karnataka", "tamil nadu", "telangana", "west bengal", "bihar", "kerala", "madhya pradesh", "assam", "odisha", "himachal pradesh", "uttarakhand", "andhra pradesh"]:
                        entities["country"] = "India"
                    else:
                        entities["country"] = "United States"
                    break

        # 4. Check country explicitly if mentioned
        if "india" in raw_text or "indian" in raw_text:
            entities["country"] = "India"
            if not entities["state"]:
                entities["state"] = "Punjab" # Default major state if unspecified
                entities["county"] = "Ludhiana"
                entities["place_name"] = "India (Punjab Basin)"

        # 5. Match Aquifers
        for aq_key, aq_full in KNOWN_AQUIFERS.items():
            if aq_key in raw_text:
                entities["aquifer"] = aq_full
                if not entities["place_name"]:
                    entities["place_name"] = aq_full
                break
                
        return entities
