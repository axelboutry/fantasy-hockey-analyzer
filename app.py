from pathlib import Path
import os
import re
import json
import datetime
import unicodedata
import requests
import random
import numpy as np
import pandas as pd
import streamlit as st
import difflib

st.set_page_config(
    page_title="NHL Player & Draft Analyzer", page_icon="🏒", layout="wide"
)

# --- CUSTOM CSS FOR POLISHED UI ---
st.markdown(
    '''
    <style>
    .metric-card {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .trade-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .math-box {
        background-color: #1a202c;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4a5568;
        margin-bottom: 20px;
    }
    .delta-positive {
        color: #51cf66;
        font-weight: bold;
    }
    .delta-negative {
        color: #ff6b6b;
        font-weight: bold;
    }
    .delta-neutral {
        color: #adb5bd;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

# --- HELPER FUNCTIONS ---
def clean_player_name(name):
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = re.sub(
        r"\s+(Jr\.?|Sr\.?|II|III|IV)$", "", name, flags=re.IGNORECASE
    )
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = name.strip().lower()
    
    aliases = {
        "joel daccord": "joey daccord",
        "mitchell marner": "mitch marner",
        "tim stuetzle": "tim stutzle",
        "zachary hyman": "zach hyman",
        "mathew dumba": "matt dumba"
    }
    return aliases.get(name, name)

def get_unique_clean_name(name, pos):
    c_name = clean_player_name(name)
    if c_name in ["elias pettersson", "sebastian aho"]:
        if pd.isna(pos):
            return f"{c_name} (f)"
        if "d" in str(pos).lower():
            return f"{c_name} (d)"
        else:
            return f"{c_name} (f)"
    return c_name

def safe_rename_player(df):
    if df.empty:
        return df
    if "skaterFullName" in df.columns:
        df["Player"] = df["skaterFullName"]
    elif "goalieFullName" in df.columns:
        df["Player"] = df["goalieFullName"]
    elif "lastName" in df.columns and "firstName" in df.columns:
        df["Player"] = df["firstName"] + " " + df["lastName"]
    elif "Player" not in df.columns:
        df["Player"] = "Unknown Player"
    return df

def get_tier_score(val, thresholds, reverse=False):
    if reverse:
        for threshold, score in thresholds:
            if val <= threshold:
                return score
        return 1.0
    else:
        for threshold, score in thresholds:
            if val >= threshold:
                return score
        return 1.0

def calculate_skater_rating(df, pace_82=False):
    if df.empty:
        return df
    d = df.copy()
    for col in ["GP", "G", "A", "PPP", "S", "+/-", "Hits", "BkS"]:
        if col not in d.columns:
            d[col] = 0.0
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    g_t = [(44, 10.0), (38, 9.0), (33, 8.0), (29, 7.0), (25, 6.0), (20, 5.0), (15, 4.0), (10, 3.0), (5, 2.0), (0, 1.0)]
    a_t = [(60, 10.0), (52, 9.0), (45, 8.0), (38, 7.0), (32, 6.0), (26, 5.0), (20, 4.0), (14, 3.0), (8, 2.0), (0, 1.0)]
    ppp_t = [(35, 10.0), (30, 9.0), (25, 8.0), (20, 7.0), (16, 6.0), (12, 5.0), (8, 4.0), (5, 3.0), (2, 2.0), (0, 1.0)]
    s_t = [(250, 10.0), (220, 9.0), (190, 8.0), (165, 7.0), (140, 6.0), (115, 5.0), (90, 4.0), (65, 3.0), (40, 2.0), (0, 1.0)]
    pm_t = [(30, 10.0), (20, 9.0), (12, 8.0), (6, 7.0), (1, 6.0), (-3, 5.0), (-10, 4.0), (-18, 3.0), (-28, 2.0), (-999, 1.0)]
    hits_t = [(220, 10.0), (180, 9.0), (140, 8.0), (110, 7.0), (85, 6.0), (65, 5.0), (45, 4.0), (25, 3.0), (10, 2.0), (0, 1.0)]
    bks_t = [(150, 10.0), (125, 9.0), (100, 8.0), (80, 7.0), (65, 6.0), (50, 5.0), (35, 4.0), (20, 3.0), (10, 2.0), (0, 1.0)]

    scores = []
    for _, row in d.iterrows():
        games = max(row["GP"], 1.0)
        pace_multiplier = 82.0 / games if games < 82 else 1.0

        g_score = get_tier_score(row["G"] * pace_multiplier, g_t)
        a_score = get_tier_score(row["A"] * pace_multiplier, a_t)
        ppp_score = get_tier_score(row["PPP"] * pace_multiplier, ppp_t)
        s_score = get_tier_score(row["S"] * pace_multiplier, s_t)
        pm_score = get_tier_score(row["+/-"], pm_t)
        hits_score = get_tier_score(row["Hits"] * pace_multiplier, hits_t)
        bks_score = get_tier_score(row["BkS"] * pace_multiplier, bks_t)

        avg_tier_10 = np.mean([g_score, a_score, ppp_score, s_score, pm_score, hits_score, bks_score])
        rating_out_of_7 = 1.0 + (avg_tier_10 - 1.0) * (6.0 / 9.0)
        
        durability_factor = min(1.0, games / 82.0) if not pace_82 else 1.0
        final_rating = 1.0 + (rating_out_of_7 - 1.0) * max(0.3, durability_factor)
        
        scores.append(round(np.clip(final_rating, 1.0, 7.0), 2))

    d["Overall_Rating"] = scores
    d["Position_Type"] = "Skater"
    return d

def calculate_goalie_rating(df, pace_82=False):
    if df.empty:
        return df
    g = df.copy()
    for col in ["GP", "W", "SV%", "GAA", "SO"]:
        if col not in g.columns:
            g[col] = 0.0
        else:
            g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)

    w_t = [(35, 10.0), (30, 9.0), (26, 8.0), (22, 7.0), (18, 6.0), (14, 5.0), (10, 4.0), (6, 3.0), (3, 2.0), (0, 1.0)]
    gaa_t = [(2.00, 10.0), (2.15, 9.0), (2.30, 8.0), (2.45, 7.0), (2.60, 6.0), (2.75, 5.0), (2.95, 4.0), (3.15, 3.0), (3.40, 2.0), (9.99, 1.0)]
    sv_t = [(0.925, 10.0), (0.920, 9.0), (0.915, 8.0), (0.910, 7.0), (0.905, 6.0), (0.900, 5.0), (0.895, 4.0), (0.890, 3.0), (0.880, 2.0), (0.0, 1.0)]
    so_t = [(7, 10.0), (6, 9.0), (5, 8.0), (4, 7.0), (3, 6.0), (2, 5.0), (1, 4.0), (0, 2.0)]

    scores = []
    for _, row in g.iterrows():
        games = max(row["GP"], 1.0)
        pace_multiplier = 55.0 / games if (games < 55 and pace_82) else 1.0
        
        w_score = get_tier_score(row["W"] * pace_multiplier, w_t)
        gaa_score = get_tier_score(row["GAA"], gaa_t, reverse=True)
        sv_score = get_tier_score(row["SV%"], sv_t)
        so_score = get_tier_score(row["SO"] * pace_multiplier, so_t)

        avg_tier_10 = np.mean([w_score, gaa_score, sv_score, so_score])
        rating_out_of_4 = 1.0 + (avg_tier_10 - 1.0) * (3.0 / 9.0)
        
        workload_factor = min(1.0, games / 55.0) if not pace_82 else 1.0
        final_rating = 1.0 + (rating_out_of_4 - 1.0) * max(0.3, workload_factor)
        
        scores.append(round(np.clip(final_rating, 1.0, 4.0), 2))

    g["Overall_Rating"] = scores
    g["Position_Type"] = "Goalie"
    return g

def get_style_for_stat(val, col_name, is_goalie=False, pace_mult=1.0):
    if pd.isna(val) or not isinstance(val, (int, float)):
        return ""
    
    paced_val = val * pace_mult
        
    if is_goalie:
        tier_maps = {
            "W": [(35, 10), (30, 9), (26, 8), (22, 7), (18, 6), (14, 5), (10, 4), (6, 3), (3, 2), (0, 1)],
            "SV%": [(0.925, 10), (0.920, 9), (0.915, 8), (0.910, 7), (0.905, 6), (0.900, 5), (0.895, 4), (0.890, 3), (0.880, 2), (0.0, 1)],
            "SO": [(7, 10), (6, 9), (5, 8), (4, 7), (3, 6), (2, 5), (1, 4), (0, 2)]
        }
        if col_name in ["W", "SO"]:
            for threshold, lvl in tier_maps[col_name]:
                if paced_val >= threshold:
                    if lvl >= 7: return "color: #51cf66; font-weight: bold;"
                    elif lvl <= 3: return "color: #ff6b6b;"
                    return ""
        elif col_name == "SV%":
            for threshold, lvl in tier_maps["SV%"]:
                if val >= threshold:
                    if lvl >= 7: return "color: #51cf66; font-weight: bold;"
                    elif lvl <= 3: return "color: #ff6b6b;"
                    return ""
        elif col_name == "GAA":
            gaa_t = [(2.00, 10), (2.15, 9), (2.30, 8), (2.45, 7), (2.60, 6), (2.75, 5), (2.95, 4), (3.15, 3), (3.40, 2), (9.99, 1)]
            for threshold, lvl in gaa_t:
                if val <= threshold:
                    if lvl >= 7: return "color: #51cf66; font-weight: bold;"
                    elif lvl <= 3: return "color: #ff6b6b;"
                    return ""
    else:
        tier_maps = {
            "G": [(44, 10), (38, 9), (33, 8), (29, 7), (25, 6), (20, 5), (15, 4), (10, 3), (5, 2), (0, 1)],
            "A": [(60, 10), (52, 9), (45, 8), (38, 7), (32, 6), (26, 5), (20, 4), (14, 3), (8, 2), (0, 1)],
            "PPP": [(35, 10), (30, 9), (25, 8), (20, 7), (16, 6), (12, 5), (8, 4), (5, 3), (2, 2), (0, 1)],
            "S": [(250, 10), (220, 9), (190, 8), (165, 7), (140, 6), (115, 5), (90, 4), (65, 3), (40, 2), (0, 1)],
            "+/-": [(30, 10), (20, 9), (12, 8), (6, 7), (1, 6), (-3, 5), (-10, 4), (-18, 3), (-28, 2), (-999, 1)],
            "Hits": [(220, 10), (180, 9), (140, 8), (110, 7), (85, 6), (65, 5), (45, 4), (25, 3), (10, 2), (0, 1)],
            "BkS": [(150, 10), (125, 9), (100, 8), (80, 7), (65, 6), (50, 5), (35, 4), (20, 3), (10, 2), (0, 1)]
        }
        if col_name in tier_maps:
            check_val = val if col_name == "+/-" else paced_val
            for threshold, lvl in tier_maps[col_name]:
                if check_val >= threshold:
                    if lvl >= 7: return "color: #51cf66; font-weight: bold;"
                    elif lvl <= 3: return "color: #ff6b6b;"
                    return ""
    return ""

def get_style_for_rating(val, is_goalie=False):
    cutoff_high = 3.2 if is_goalie else 5.2
    cutoff_low = 2.0 if is_goalie else 3.5
    if val >= cutoff_high:
        return "color: #51cf66; font-weight: bold;"
    elif val <= cutoff_low:
        return "color: #ff6b6b;"
    return "color: #adb5bd;"

# --- PERSISTENT DATA HANDLERS ---
OVERRIDE_FILE = "roster_overrides.json"
INJURY_FILE = "injury_log.json"

def load_overrides():
    if os.path.exists(OVERRIDE_FILE):
        try:
            with open(OVERRIDE_FILE, "r") as f:
                data = json.load(f)
                normalized = {}
                for k, v in data.items():
                    if isinstance(v, str):
                        v = v.replace("Let's Do That Hockey", "Let’s Do That Hockey")
                        v = v.replace("Seeing Ghosts", "Return of the Mac")
                    normalized[k] = v
                return normalized
        except Exception:
            pass
    return {}

def save_overrides(ov_dict):
    try:
        with open(OVERRIDE_FILE, "w") as f:
            json.dump(ov_dict, f, indent=4)
    except Exception:
        pass

def load_injuries():
    if os.path.exists(INJURY_FILE):
        try:
            with open(INJURY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_injuries(inj_dict):
    try:
        with open(INJURY_FILE, "w") as f:
            json.dump(inj_dict, f, indent=4)
    except Exception:
        pass

if "roster_overrides" not in st.session_state:
    st.session_state["roster_overrides"] = load_overrides()

local_injuries = load_injuries()

# --- DRAFT LOAD & MAP GENERATION ---
draft_df = pd.DataFrame()
for draft_fname in ["draft_results", "draft_results.csv"]:
    if os.path.exists(draft_fname):
        try:
            draft_df = pd.read_csv(draft_fname)
            if "Fantasy Team" in draft_df.columns:
                draft_df["Fantasy Team"] = draft_df["Fantasy Team"].replace(
                    {
                        "Let’s Do T...": "Let’s Do That Hockey", 
                        "Let's Do T...": "Let’s Do That Hockey",
                        "Let's Do That Hockey": "Let’s Do That Hockey",
                        "Seeing Ghosts": "Return of the Mac"
                    }
                )
            break
        except Exception:
            pass

draft_players_list = []
if not draft_df.empty and "Player" in draft_df.columns:
    for _, row in draft_df.iterrows():
        raw_p = str(row["Player"])
        match = re.search(r"^(.*?)\((.*?)\s*-\s*(.*?)\)$", raw_p)
        if match:
            name, team, pos = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
        else:
            name, team, pos = raw_p, "FA", "C"
        
        c_name = get_unique_clean_name(name, pos)
        display_name = name
        if c_name.endswith("(d)") and not name.endswith("(D)"): display_name += " (D)"
        if c_name.endswith("(f)") and not name.endswith("(F)"): display_name += " (F)"
        
        draft_players_list.append({
            "Player": display_name,
            "Clean_Player": c_name,
            "Team": team,
            "Pos": pos,
            "Fantasy_Team": row.get("Fantasy Team", "Waiver Wire / Free Agent")
        })
draft_db = pd.DataFrame(draft_players_list)

drafted_player_map = {}
acquisition_map = {}
position_map = {}
real_team_map = {}

if not draft_db.empty:
    for _, row in draft_db.iterrows():
        c_name = row["Clean_Player"]
        f_team = row["Fantasy_Team"]
        drafted_player_map[c_name] = f_team
        acquisition_map[c_name] = "🎯 Drafted"
        position_map[c_name] = str(row["Pos"])
        real_team_map[c_name] = str(row["Team"])

for c_name, assigned_team in st.session_state["roster_overrides"].items():
    drafted_player_map[c_name] = assigned_team
    if assigned_team == "Waiver Wire / Free Agent":
        acquisition_map[c_name] = "🗑️ Dropped"
    else:
        acquisition_map[c_name] = "🔄 Trade/Waiver"

# --- BACKGROUND FETCHERS ---
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_base_teams_and_positions():
    base_info = {}
    for season in ["20262027", "20252026"]:
        try:
            for endpoint in ["skater/summary", "goalie/summary"]:
                url = f"https://api.nhle.com/stats/rest/en/{endpoint}?isAggregate=false&isGame=false&limit=-1&cayenneExp=seasonId={season}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    for p in resp.json().get("data", []):
                        name_key = "skaterFullName" if "skater" in endpoint else "goalieFullName"
                        pos = p.get("positionCode", "G")
                        c_name = get_unique_clean_name(p.get(name_key, ""), pos)
                        pos_trans = {"L": "LW", "R": "RW", "C": "C", "D": "D"}.get(pos, pos)
                        if c_name and c_name not in base_info:
                            base_info[c_name] = {
                                "Team": p.get("teamAbbrevs", "FA"),
                                "Pos": pos_trans
                            }
        except Exception:
            pass
        if base_info:
            break
    return base_info    

# --- BULLETPROOF RAW TEXT SCRAPER ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_injury_text_dump():
    combined_text = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    }
    
    # Hit multiple sources to guarantee we don't miss hidden LTIR players
    urls = [
        "https://www.cbssports.com/nhl/injuries/",
        "https://www.espn.com/nhl/injuries",
        "https://www.foxsports.com/nhl/injuries"
    ]
    
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            # Rip out all code logic and styling
            html = re.sub(r'<script.*?</script>', ' ', resp.text, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
            # Strip remaining HTML brackets to leave pure raw text
            text = re.sub(r'<[^>]+>', ' ', html)
            # Compress extra spaces and lowercase everything
            text = re.sub(r'\s+', ' ', text).lower()
            combined_text += text + " "
        except Exception:
            continue
            
    return combined_text

def get_injury_tag(clean_name, combined_text):
    if not combined_text or not isinstance(combined_text, str): 
        return ""
        
    search_name = clean_name.replace(" (d)", "").replace(" (f)", "").strip().lower()
    parts = search_name.split()
    if len(parts) < 2: return ""
    
    first_name = parts[0]
    last_name = parts[-1]
    first_init = first_name[0]
    
    # Create different spelling combinations to catch Yahoo vs ESPN typos (Aleksander vs Alexander)
    name_variants = [
        f"{first_name} {last_name}",
        f"{first_init}. {last_name}",
        f"{first_init} {last_name}",
        f"{last_name}, {first_name}"
    ]
    if first_name == "aleksander":
        name_variants.append(f"alexander {last_name}")
        name_variants.append(f"barkov, alexander")
        
    # Search the massive wall of text for the player's name
    for variant in name_variants:
        matches = [m.start() for m in re.finditer(re.escape(variant), combined_text)]
        for match_idx in matches:
            # If the name is found, grab the 120 characters immediately surrounding it
            start = max(0, match_idx - 60)
            end = min(len(combined_text), match_idx + 120)
            window = combined_text[start:end]
            
            # If any injury keyword appears in that same window, tag them
            if bool(re.search(r'\b(out|ir|injured reserve|ltir|ir-lt|ir-nr|non-roster)\b', window)):
                return "🚨 OUT"
            if bool(re.search(r'\b(day-to-day|dtd|day to day)\b', window)):
                return "🏥 DTD"
            
    return ""


def inject_missing_draft_players(df, is_goalie=False):
    if df.empty: 
        return df
    
    existing = set(df.get("Clean_Player", pd.Series()).dropna().tolist())
    missing_records = []

    if not draft_db.empty:
        for _, row in draft_db.iterrows():
            if row["Clean_Player"] not in existing:
                is_g = "G" in str(row["Pos"])
                if (is_goalie and is_g) or (not is_goalie and not is_g):
                    missing_records.append(row.to_dict())
                    existing.add(row["Clean_Player"])

    bg_info = fetch_base_teams_and_positions()
    for c_name, t_team in st.session_state["roster_overrides"].items():
        if c_name not in existing:
            p_info = bg_info.get(c_name, {})
            pos = p_info.get("Pos", "C")
            is_g = pos == "G"
            if (is_goalie and is_g) or (not is_goalie and not is_g):
                missing_records.append({
                    "Clean_Player": c_name,
                    "Player": c_name.title().replace(" (D)", "").replace(" (F)", "") + (" (D)" if c_name.endswith("(d)") else (" (F)" if c_name.endswith("(f)") else "")),
                    "Team": p_info.get("Team", "FA"),
                    "Pos": pos,
                    "Fantasy_Team": t_team
                })
                existing.add(c_name)

    if not missing_records:
        return df
        
    missing_df = pd.DataFrame(missing_records)
    for col in df.columns:
        if col not in missing_df.columns:
            if col in ["GP", "G", "A", "PPP", "S", "+/-", "Hits", "BkS", "W", "SV%", "GAA", "SO", "Overall_Rating"]:
                missing_df[col] = 0.0
            elif col == "Prospect_Status":
                missing_df[col] = "Veteran"
            else:
                missing_df[col] = np.nan
                
    return pd.concat([df, missing_df], ignore_index=True)

# --- NHL API LIVE DATA PIPELINE ---
@st.cache_data(show_spinner="Fetching Live NHL Stats...")
def load_all_seasons(pace_82=False):
    seasons = ["20262027", "20252026", "20242025"]
    season_display_map = {
        "20262027": "2026-27",
        "20252026": "2025-26", 
        "20242025": "2024-25"
    }
    skater_data, goalie_data = {}, {}

    def fetch_api(endpoint, season):
        url = f"https://api.nhle.com/stats/rest/en/{endpoint}?isAggregate=false&isGame=false&start=0&limit=-1&factCayenneExp=gamesPlayed>=0&cayenneExp=gameTypeId=2%20and%20seasonId={season}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

    for season in seasons:
        display_season = season_display_map[season]
        df_score = fetch_api("skater/summary", season)
        df_misc = fetch_api("skater/realtime", season)
        
        if not df_score.empty and not df_misc.empty:
            df_score = safe_rename_player(df_score)
            df_score = df_score.rename(columns={
                "teamAbbrevs": "Team", "positionCode": "Pos", "gamesPlayed": "GP",
                "goals": "G", "assists": "A", "ppPoints": "PPP", "shots": "S", "plusMinus": "+/-"
            })
            df_misc = safe_rename_player(df_misc)
            df_misc = df_misc.rename(columns={"hits": "Hits", "blockedShots": "BkS"})
            
            misc_cols_to_keep = ["Clean_Player"]
            if "Hits" in df_misc.columns: misc_cols_to_keep.append("Hits")
            if "BkS" in df_misc.columns: misc_cols_to_keep.append("BkS")
                
            df_score["Clean_Player"] = df_score.apply(lambda r: get_unique_clean_name(r["Player"], r["Pos"]), axis=1)
            df_misc["Clean_Player"] = df_misc.apply(lambda r: get_unique_clean_name(r["Player"], r.get("positionCode", "F")), axis=1)
            df_score["Player"] = df_score.apply(lambda r: r["Player"] + " (D)" if get_unique_clean_name(r["Player"], r["Pos"]).endswith("(d)") else (r["Player"] + " (F)" if get_unique_clean_name(r["Player"], r["Pos"]).endswith("(f)") else r["Player"]), axis=1)
            
            df_comb = pd.merge(df_score, df_misc[misc_cols_to_keep], on="Clean_Player", how="outer")
            df_comb = inject_missing_draft_players(df_comb, is_goalie=False)
            df_comb["Prospect_Status"] = "Veteran"
            skater_data[display_season] = calculate_skater_rating(df_comb, pace_82)
            
        df_goalie = fetch_api("goalie/summary", season)
        if not df_goalie.empty:
            df_goalie = safe_rename_player(df_goalie)
            df_goalie = df_goalie.rename(columns={
                "teamAbbrevs": "Team", "gamesPlayed": "GP", "wins": "W",
                "savePct": "SV%", "goalsAgainstAverage": "GAA", "shutouts": "SO"
            })
            df_goalie["Clean_Player"] = df_goalie["Player"].apply(clean_player_name)
            df_goalie = inject_missing_draft_players(df_goalie, is_goalie=True)
            goalie_data[display_season] = calculate_goalie_rating(df_goalie, pace_82)
            
    return skater_data, goalie_data

@st.cache_data(show_spinner="Fetching Live Streak Stats...")
def fetch_streak_data(start_d, end_d, fetch_goalies=False, pace_82=False):
    base_url = "https://api.nhle.com/stats/rest/en"
    exp = f'gameDate>="{start_d}"%20and%20gameDate<="{end_d}"%20and%20gameTypeId=2'
    
    if not fetch_goalies:
        url_score = f"{base_url}/skater/summary?isAggregate=true&isGame=true&start=0&limit=-1&factCayenneExp=gamesPlayed>=0&cayenneExp={exp}"
        url_misc = f"{base_url}/skater/realtime?isAggregate=true&isGame=true&start=0&limit=-1&factCayenneExp=gamesPlayed>=0&cayenneExp={exp}"
        try:
            score_data = requests.get(url_score, timeout=15).json().get("data", [])
            misc_data = requests.get(url_misc, timeout=15).json().get("data", [])
        except:
            return pd.DataFrame()
            
        df_score = pd.DataFrame(score_data)
        df_misc = pd.DataFrame(misc_data)
        if df_score.empty: return pd.DataFrame()
        
        df_score = safe_rename_player(df_score)
        df_score = df_score.rename(columns={
            "teamAbbrevs": "Team", "positionCode": "Pos", "gamesPlayed": "GP", 
            "goals": "G", "assists": "A", "ppPoints": "PPP", "shots": "S", "plusMinus": "+/-"
        })
        
        if not df_misc.empty:
            df_misc = safe_rename_player(df_misc)
            df_misc = df_misc.rename(columns={"hits": "Hits", "blockedShots": "BkS"})
            misc_cols = ["Clean_Player"]
            if "Hits" in df_misc.columns: misc_cols.append("Hits")
            if "BkS" in df_misc.columns: misc_cols.append("BkS")
            
            df_score["Clean_Player"] = df_score.apply(lambda r: get_unique_clean_name(r["Player"], r["Pos"]), axis=1)
            df_misc["Clean_Player"] = df_misc.apply(lambda r: get_unique_clean_name(r["Player"], r.get("positionCode", "F")), axis=1)
            df_score["Player"] = df_score.apply(lambda r: r["Player"] + " (D)" if get_unique_clean_name(r["Player"], r["Pos"]).endswith("(d)") else (r["Player"] + " (F)" if get_unique_clean_name(r["Player"], r["Pos"]).endswith("(f)") else r["Player"]), axis=1)
            df_comb = pd.merge(df_score, df_misc[misc_cols], on="Clean_Player", how="left")
        else:
            df_score["Clean_Player"] = df_score.apply(lambda r: get_unique_clean_name(r["Player"], r["Pos"]), axis=1)
            df_score["Player"] = df_score.apply(lambda r: r["Player"] + " (D)" if get_unique_clean_name(r["Player"], r["Pos"]).endswith("(d)") else (r["Player"] + " (F)" if get_unique_clean_name(r["Player"], r["Pos"]).endswith("(f)") else r["Player"]), axis=1)
            df_comb = df_score
            
        df_comb = inject_missing_draft_players(df_comb, is_goalie=False)
        return calculate_skater_rating(df_comb, pace_82=pace_82)
    else:
        url_goalie = f"{base_url}/goalie/summary?isAggregate=true&isGame=true&start=0&limit=-1&factCayenneExp=gamesPlayed>=0&cayenneExp={exp}"
        try:
            g_data = requests.get(url_goalie, timeout=15).json().get("data", [])
        except:
            return pd.DataFrame()
            
        df_goalie = pd.DataFrame(g_data)
        if df_goalie.empty: return pd.DataFrame()
        
        df_goalie = safe_rename_player(df_goalie)
        df_goalie = df_goalie.rename(columns={
            "teamAbbrevs": "Team", "gamesPlayed": "GP",
            "wins": "W", "savePct": "SV%", "goalsAgainstAverage": "GAA", "shutouts": "SO"
        })
        df_goalie["Clean_Player"] = df_goalie["Player"].apply(clean_player_name)
        df_goalie = inject_missing_draft_players(df_goalie, is_goalie=True)
        return calculate_goalie_rating(df_goalie, pace_82=pace_82)

# --- WEIGHTED MULTI-SEASON AGGREGATION ---
WEIGHTS_MAP = {"2025-26": 0.8, "2024-25": 0.5, "2023-24": 0.2}

@st.cache_data
def aggregate_multi_season_skaters(sk_dict, pace_82=False):
    if not sk_dict: return pd.DataFrame()
    weighted_dfs = []
    for season, df in sk_dict.items():
        if df.empty: continue
        w = WEIGHTS_MAP.get(season, 1.0)
        temp = df.copy()
        for col in ["GP", "G", "A", "PPP", "S", "+/-", "Hits", "BkS"]:
            if col in temp.columns:
                temp[col] = pd.to_numeric(temp[col], errors="coerce").fillna(0.0) * w
        temp["Season_Weight"] = w
        weighted_dfs.append(temp)
    all_df = pd.concat(weighted_dfs, ignore_index=True)
    if all_df.empty: return all_df
    
    agg_rules = {"Player": "first", "Team": "last", "Pos": "first", "GP": "sum", "G": "sum", "A": "sum",
                 "PPP": "sum", "S": "sum" if "S" in all_df.columns else "first", "+/-": "sum", "Hits": "sum",
                 "BkS": "sum", "Season_Weight": "sum"}
    actual_rules = {k: v for k, v in agg_rules.items() if k in all_df.columns}
    combined = all_df.groupby("Clean_Player", as_index=False).agg(actual_rules)
    for col in ["GP", "G", "A", "PPP", "S", "+/-", "Hits", "BkS"]:
        if col in combined.columns:
            combined[col] = combined[col] / combined["Season_Weight"]
    combined["Season"] = "Multi-Season Combined (80/50/20)"
    return calculate_skater_rating(combined, pace_82)

@st.cache_data
def aggregate_multi_season_goalies(g_dict, pace_82=False):
    if not g_dict: return pd.DataFrame()
    weighted_dfs = []
    for season, df in g_dict.items():
        if df.empty: continue
        w = WEIGHTS_MAP.get(season, 1.0)
        temp = df.copy()
        for col in ["GP", "W", "SO", "SV%", "GAA"]:
            if col in temp.columns:
                temp[col] = pd.to_numeric(temp[col], errors="coerce").fillna(0.0) * w
        temp["Season_Weight"] = w
        if "Clean_Player" not in temp.columns and "Player" in temp.columns:
            temp["Clean_Player"] = temp["Player"].apply(clean_player_name)
        weighted_dfs.append(temp)
    all_df = pd.concat(weighted_dfs, ignore_index=True)
    if all_df.empty: return all_df
    
    agg_rules = {"Player": "first", "Team": "last", "Pos": "first", "GP": "sum",
                 "W": "sum" if "W" in all_df.columns else "first", "SO": "sum", "SV%": "sum", "GAA": "sum", "Season_Weight": "sum"}
    actual_rules = {k: v for k, v in agg_rules.items() if k in all_df.columns}
    combined = all_df.groupby("Clean_Player", as_index=False).agg(actual_rules)
    for col in ["GP", "W", "SO", "SV%", "GAA"]:
        if col in combined.columns:
            combined[col] = combined[col] / combined["Season_Weight"]
    combined["Season"] = "Multi-Season Combined (80/50/20)"
    return calculate_goalie_rating(combined, pace_82)

# --- NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "App Mode",
    options=[
        "📊 League & Waiver Hub",
        "🤝 Trade & Transaction Hub",
        "📐 Math & Formula Inspector",
        "📰 Weekly Gazette / League Journal",
        "🔮 Oracle Projections (Beat the Val Model)",
    ],
)

# --- DATA SETTINGS IN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Data & Season Settings")
view_mode = st.sidebar.radio("View Scope", options=["This Season (2026-27)", "Single Season", "Multi-Season Combined (80/50/20)"])

selected_mode = None
time_filter = None

if view_mode == "This Season (2026-27)":
    time_filter = st.sidebar.selectbox("Timeframe", ["Season to Date", "Last 7 Days", "Last 14 Days", "Last 30 Days"])
    selected_mode = f"2026-27 ({time_filter})"
elif view_mode == "Single Season":
    available_seasons = ["2025-26", "2024-25", "2023-24"]
    selected_mode = st.sidebar.selectbox("Select Season", options=available_seasons, index=0)
else:
    selected_mode = "Multi-Season Combined (80/50/20)"

apply_82_pace = st.sidebar.checkbox("🚀 82-Game Pace (No Durability Penalty)", value=False)
if view_mode == "This Season (2026-27)" and time_filter != "Season to Date":
    apply_82_pace = True
    st.sidebar.caption("*(82-Game Pace is locked ON for short streak timeframes)*")

if view_mode == "This Season (2026-27)":
    today = datetime.date.today()
    if time_filter == "Last 7 Days":
        start_date = today - datetime.timedelta(days=7)
    elif time_filter == "Last 14 Days":
        start_date = today - datetime.timedelta(days=14)
    elif time_filter == "Last 30 Days":
        start_date = today - datetime.timedelta(days=30)
    else:
        start_date = datetime.date(2026, 9, 29)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")
    st.sidebar.caption(f"**Data Window:**\n{start_str} to {end_str}")
    
    current_skaters = fetch_streak_data(start_str, end_str, fetch_goalies=False, pace_82=apply_82_pace)
    current_goalies = fetch_streak_data(start_str, end_str, fetch_goalies=True, pace_82=apply_82_pace)
else:
    skater_data, goalie_data = load_all_seasons(apply_82_pace)
    master_skaters_df = aggregate_multi_season_skaters(skater_data, apply_82_pace)
    master_goalies_df = aggregate_multi_season_goalies(goalie_data, apply_82_pace)
    current_skaters = skater_data.get(selected_mode, pd.DataFrame()) if view_mode == "Single Season" else master_skaters_df.copy()
    current_goalies = goalie_data.get(selected_mode, pd.DataFrame()) if view_mode == "Single Season" else master_goalies_df.copy()

nhl_base_info = fetch_base_teams_and_positions()

# LOAD THE NEW INDESTRUCTIBLE RAW TEXT SCRAPER
injury_text_dump = fetch_injury_text_dump()

def get_best_value(c_name, default, maps):
    for m in maps:
        val = m.get(c_name) if isinstance(m, dict) else m
        if val and str(val).lower() not in ["none", "nan", ""]:
            return val
    return default

def get_manual_injury_tag(clean_name, local_inj):
    for inj_player, details in local_inj.items():
        if clean_player_name(inj_player) == clean_name.replace(" (d)", "").replace(" (f)", ""):
            status = details.get("Status", "IR")
            return f"🚑 [{status}]"
    return ""

if not current_skaters.empty:
    current_skaters["Player"] = current_skaters.apply(
        lambda row: f"{row['Player']} {get_manual_injury_tag(row['Clean_Player'], local_injuries)} {get_injury_tag(row['Clean_Player'], injury_text_dump)}".strip(), 
        axis=1
    )
    current_skaters["League_Status"] = current_skaters["Clean_Player"].apply(lambda x: drafted_player_map.get(x, "Waiver Wire / Free Agent"))
    current_skaters["Fantasy_Team"] = current_skaters["League_Status"]
    current_skaters["Acquired"] = current_skaters["Clean_Player"].apply(lambda x: acquisition_map.get(x, "-"))
    current_skaters["Team"] = current_skaters.apply(
        lambda row: get_best_value(row["Clean_Player"], "FA", [
            real_team_map, nhl_base_info.get(row["Clean_Player"], {}).get("Team"), row.get("Team")
        ]), axis=1
    )
    current_skaters["Pos"] = current_skaters.apply(
        lambda row: get_best_value(row["Clean_Player"], "C", [
            position_map, nhl_base_info.get(row["Clean_Player"], {}).get("Pos"), row.get("Pos")
        ]), axis=1
    )
    def get_pos_bonus(pos_str):
        if not isinstance(pos_str, str): return 0.0
        count = len([p for p in pos_str.split(',') if p.strip()])
        return max(0, count - 1) * 0.20
    current_skaters["Overall_Rating"] = current_skaters["Overall_Rating"] + current_skaters["Pos"].apply(get_pos_bonus)

if not current_goalies.empty:
    if "Clean_Player" not in current_goalies.columns:
        if "Player" in current_goalies.columns:
            current_goalies["Clean_Player"] = current_goalies["Player"].apply(clean_player_name)
        elif "goalieFullName" in current_goalies.columns:
            current_goalies["Clean_Player"] = current_goalies["goalieFullName"].apply(clean_player_name)
            
    if "Pos" not in current_goalies.columns:
        current_goalies["Pos"] = "G"

    # Set team mapping first before formatting Player string
    current_goalies["Team"] = current_goalies.apply(
        lambda row: get_best_value(row["Clean_Player"], "FA", [
            real_team_map, nhl_base_info.get(row["Clean_Player"], {}).get("Team"), row.get("Team")
        ]), axis=1
    )
    
    current_goalies["League_Status"] = current_goalies["Clean_Player"].apply(
        lambda x: drafted_player_map.get(x, "Waiver Wire / Free Agent")
    )
    current_goalies["Fantasy_Team"] = current_goalies["League_Status"]
    current_goalies["Acquired"] = current_goalies["Clean_Player"].apply(
        lambda x: acquisition_map.get(x, "-")
    )

    # Append injury tags to display name
    current_goalies["Player"] = current_goalies.apply(
        lambda row: f"{row['Player']} {get_manual_injury_tag(row['Clean_Player'], local_injuries)} {get_injury_tag(row['Clean_Player'], injury_text_dump)}".strip(), 
        axis=1
    )

# Unified pool
comb_skaters = current_skaters.copy()
comb_skaters["Position_Type"] = "Skater"

comb_goalies = current_goalies.copy()
comb_goalies["Position_Type"] = "Goalie"

unified_pool = pd.concat([comb_skaters, comb_goalies], ignore_index=True, sort=False)

if not unified_pool.empty:
    team_series = unified_pool["Team"] if "Team" in unified_pool.columns else pd.Series("FA", index=unified_pool.index)
    pos_series = unified_pool["Pos"].fillna("G") if "Pos" in unified_pool.columns else pd.Series("G", index=unified_pool.index)
    unified_pool["Display_Label"] = (
        unified_pool["Player"].astype(str)
        + " ("
        + team_series.fillna("FA").astype(str)
        + " - "
        + pos_series.astype(str)
        + ")"
    )

all_teams_in_league = sorted(list(set(draft_db["Fantasy_Team"].dropna().unique()))) if not draft_db.empty else ["Let’s Do That Hockey"]
if "Waiver Wire / Free Agent" in all_teams_in_league:
    all_teams_in_league.remove("Waiver Wire / Free Agent")
all_roster_entities = ["Waiver Wire / Free Agent"] + all_teams_in_league

# Weekly Schedule
WEEKLY_SCHEDULE = {
    "Week 1": {"start": "2026-09-29", "end": "2026-10-04"},
    "Week 2": {"start": "2026-10-05", "end": "2026-10-11"},
    "Week 3": {"start": "2026-10-12", "end": "2026-10-18"},
    "Week 4": {"start": "2026-10-19", "end": "2026-10-25"},
    "Week 5": {"start": "2026-10-26", "end": "2026-11-01"},
    "Week 6": {"start": "2026-11-02", "end": "2026-11-08"},
    "Week 7": {"start": "2026-11-09", "end": "2026-11-15"},
    "Week 8": {"start": "2026-11-16", "end": "2026-11-22"},
    "Week 9": {"start": "2026-11-23", "end": "2026-11-29"},
    "Week 10": {"start": "2026-11-30", "end": "2026-12-06"},
    "Week 11": {"start": "2026-12-07", "end": "2026-12-13"},
    "Week 12": {"start": "2026-12-14", "end": "2026-12-20"},
    "Week 13": {"start": "2026-12-21", "end": "2026-12-27"},
    "Week 14": {"start": "2026-12-28", "end": "2027-01-03"},
    "Week 15": {"start": "2027-01-04", "end": "2027-01-10"},
    "Week 16": {"start": "2027-01-11", "end": "2027-01-17"},
    "Week 17": {"start": "2027-01-18", "end": "2027-01-24"},
    "Week 18": {"start": "2027-01-25", "end": "2027-01-31"},
    "Week 19": {"start": "2027-02-01", "end": "2027-02-14", "note": "All-Star Break"},
    "Week 20": {"start": "2027-02-15", "end": "2027-02-21"},
    "Week 21": {"start": "2027-02-22", "end": "2027-02-28"},
    "Week 22": {"start": "2027-03-01", "end": "2027-03-07"},
    "Week 23": {"start": "2027-03-08", "end": "2027-03-14", "playoffs": "Quarterfinals"},
    "Week 24": {"start": "2027-03-15", "end": "2027-03-21", "playoffs": "Semifinals"}, 
    "Week 25": {"start": "2027-03-22", "end": "2027-03-28", "playoffs": "Championship"},
}


# ==========================================
# TAB 1: LEAGUE & WAIVER HUB
# ==========================================
if app_mode == "📊 League & Waiver Hub":
    st.title("📊 League & Waiver Hub")
    st.markdown("View all rosters, filter free agents, and examine draft outcomes.")

    tab1, tab2 = st.tabs(["📋 Roster & Waiver Pool", "📋 Draft Analysis"])

    with tab1:
        col_f1, col_f2 = st.columns([1.5, 1])
        with col_f1:
            available_pools = ["All Players", "My Team ('Let’s Do That Hockey')", "Waiver Wire / Free Agent"]
            if not draft_df.empty and "Fantasy Team" in draft_df.columns:
                opp_teams = sorted([t for t in draft_df["Fantasy Team"].unique() if "Let’s Do That Hockey" not in str(t)])
                available_pools.extend([f"Opponent: {t}" for t in opp_teams])
            pool_choice = st.selectbox("Select Roster Bucket", options=available_pools)
            
        with col_f2:
            st.markdown("<br>", unsafe_allow_html=True)
            show_goalies_toggle = st.checkbox("🥅 Show Goalies Pool")
            st.caption("*(Goalies display in a separate view due to stat layout)*")

        active_pool = current_goalies.copy() if show_goalies_toggle else current_skaters.copy()

        if active_pool.empty:
            st.info(f"No {'goalie' + ('s' if not show_goalies_toggle else '')} data available for {selected_mode}.")
        else:
            if pool_choice == "My Team ('Let’s Do That Hockey')":
                active_pool = active_pool[active_pool["Fantasy_Team"].str.contains(r"Let[’']s Do That Hockey", case=False, na=False, regex=True)]
            elif pool_choice == "Waiver Wire / Free Agent":
                active_pool = active_pool[active_pool["Fantasy_Team"] == "Waiver Wire / Free Agent"]
            elif pool_choice.startswith("Opponent: "):
                target_opp = pool_choice.replace("Opponent: ", "").strip()
                safe_opp = target_opp.replace("'", ".").replace("’", ".")
                active_pool = active_pool[active_pool["Fantasy_Team"].str.contains(safe_opp, case=False, na=False, regex=True)]

            avg_gp = active_pool["GP"].mean() if "GP" in active_pool.columns and active_pool["GP"].mean() > 0 else 1.0
            pace_mult = (82.0 / avg_gp) if (apply_82_pace and avg_gp < 82) else 1.0

            if show_goalies_toggle:
                display_cols = [c for c in ["Player", "Team", "Pos", "Fantasy_Team", "Acquired", "GP", "Overall_Rating", "W", "GAA", "SV%", "SO"] if c in active_pool.columns]
                disp_df = active_pool[display_cols].sort_values(by="Overall_Rating", ascending=False)
                styled_df = disp_df.style.map(lambda v: get_style_for_stat(v, "W", True, pace_mult), subset=["W"]) \
                                         .map(lambda v: get_style_for_stat(v, "GAA", True, 1.0), subset=["GAA"]) \
                                         .map(lambda v: get_style_for_stat(v, "SV%", True, 1.0), subset=["SV%"]) \
                                         .map(lambda v: get_style_for_stat(v, "SO", True, pace_mult), subset=["SO"]) \
                                         .map(lambda v: get_style_for_rating(v, True), subset=["Overall_Rating"]) \
                                         .format({
                                             "GP": "{:.0f}", "W": "{:.0f}", "GAA": "{:.2f}",
                                             "SV%": "{:.3f}", "SO": "{:.0f}", "Overall_Rating": "{:.2f}"
                                         })
            else:
                display_cols = [c for c in ["Player", "Team", "Pos", "Fantasy_Team", "Acquired", "GP", "Overall_Rating", "G", "A", "+/-", "PPP", "S", "Hits", "BkS"] if c in active_pool.columns]
                disp_df = active_pool[display_cols].sort_values(by="Overall_Rating", ascending=False)
                styled_df = disp_df.style.map(lambda v: get_style_for_stat(v, "G", False, pace_mult), subset=["G"]) \
                                         .map(lambda v: get_style_for_stat(v, "A", False, pace_mult), subset=["A"]) \
                                         .map(lambda v: get_style_for_stat(v, "PPP", False, pace_mult), subset=["PPP"]) \
                                         .map(lambda v: get_style_for_stat(v, "S", False, pace_mult), subset=["S"]) \
                                         .map(lambda v: get_style_for_stat(v, "+/-", False, 1.0), subset=["+/-"]) \
                                         .map(lambda v: get_style_for_stat(v, "Hits", False, pace_mult) if "Hits" in disp_df.columns else None, subset=["Hits"] if "Hits" in disp_df.columns else []) \
                                         .map(lambda v: get_style_for_stat(v, "BkS", False, pace_mult) if "BkS" in disp_df.columns else None, subset=["BkS"] if "BkS" in disp_df.columns else []) \
                                         .map(lambda v: get_style_for_rating(v, False), subset=["Overall_Rating"]) \
                                         .format({
                                             "GP": "{:.0f}", "G": "{:.0f}", "A": "{:.0f}", "PPP": "{:.0f}",
                                             "S": "{:.0f}", "+/-": "{:.0f}", "Hits": "{:.0f}", "BkS": "{:.0f}",
                                             "Overall_Rating": "{:.2f}"
                                         })

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Overall_Rating": st.column_config.ProgressColumn(
                        "Goalie Rating (1-4)" if show_goalies_toggle else "Tiered Rating (1-7)",
                        help="Absolute tiered score out of 4 categories for goalies (or 7 for skaters)",
                        format="%.2f",
                        min_value=1.0,
                        max_value=4.0 if show_goalies_toggle else 7.4,
                    )
                },
            )

    with tab2:
        st.subheader("Draft Analysis & Roster Grades")
        if draft_df.empty:
            st.info("No draft results file found ('draft_results.csv').")
        else:
            st.dataframe(draft_df, use_container_width=True, hide_index=True)


# ==========================================
# TAB 2: TRADE & IR HUB
# ==========================================
elif app_mode == "🤝 Trade & Transaction Hub":
    st.title("🤝 Trade, Transaction & IR Hub")
    st.markdown("Execute trades, evaluate category impact, and manage injured reserve designations.")

    tab_trade, tab_ir = st.tabs(["🔄 Trade Analyzer & Execution", "🚑 Injury & IR Ledger"])

    with tab_trade:
        if unified_pool.empty:
            st.warning("No live player data available for analysis. (Check your date filters or NHL API status).")
        else:
            col_ov1, col_ov2, col_ov3 = st.columns([2, 2, 1])
            with col_ov1:
                source_team = st.selectbox("From (Source Team / Pool)", options=all_roster_entities, key="trade_source")
            with col_ov2:
                dest_options = [t for t in all_roster_entities if t != source_team]
                dest_team = st.selectbox("To (Destination)", options=dest_options, key="trade_dest")
                
            col_pick1, col_pick2, _ = st.columns([2, 2, 1])
            with col_pick1:
                source_pool = unified_pool[unified_pool["Fantasy_Team"] == source_team]
                source_player_options = sorted(source_pool["Display_Label"].dropna().unique().tolist())
                side_a_picks = st.multiselect(f"Select Player(s) to Move from {source_team}", source_player_options, key="trade_side_a")
                
            with col_pick2:
                dest_pool = unified_pool[unified_pool["Fantasy_Team"] == dest_team]
                dest_player_options = sorted(dest_pool["Display_Label"].dropna().unique().tolist())
                side_b_picks = st.multiselect(f"Select Player(s) to Move from {dest_team}", dest_player_options, key="trade_side_b")

            with col_ov3:
                st.markdown("<br>", unsafe_allow_html=True)
                btn_disabled = (len(side_a_picks) == 0 and len(side_b_picks) == 0)
                if st.button("Execute Move", disabled=btn_disabled, use_container_width=True):
                    updates_made = False
                    for p_label in side_a_picks:
                        p_name = unified_pool[unified_pool["Display_Label"] == p_label].iloc[0]["Clean_Player"]
                        st.session_state["roster_overrides"][p_name] = dest_team
                        updates_made = True
                    for p_label in side_b_picks:
                        p_name = unified_pool[unified_pool["Display_Label"] == p_label].iloc[0]["Clean_Player"]
                        st.session_state["roster_overrides"][p_name] = source_team
                        updates_made = True
                    
                    if updates_made:
                        save_overrides(st.session_state["roster_overrides"])
                        st.success("Transaction executed successfully! Rosters updated.")
                        st.rerun()

            df_a = unified_pool[unified_pool["Display_Label"].isin(side_a_picks)]
            df_b = unified_pool[unified_pool["Display_Label"].isin(side_b_picks)]

            def get_normalized_trade_value(df):
                if df.empty: return 0.0
                val = 0.0
                for _, row in df.iterrows():
                    if row["Position_Type"] == "Goalie":
                        val += 1.0 + ((row["Overall_Rating"] - 1.0) * 2.0)
                    else:
                        val += row["Overall_Rating"]
                return val

            score_a = get_normalized_trade_value(df_a)
            score_b = get_normalized_trade_value(df_b)
            net_gain_a = round(score_b - score_a, 2)
            net_gain_b = round(score_a - score_b, 2)

            st.markdown("---")
            st.subheader("Trade Value & Impact")
            
            res_col1, res_col2, res_col3 = st.columns([2, 1.2, 2])
            with res_col1:
                st.metric(label=f"Value Given Up by {source_team}", value=round(score_a, 2), delta=f"{net_gain_a:+.2f}" if net_gain_a != 0 else "Even")
            with res_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if net_gain_a > 0:
                    st.markdown(f"<h4 style='text-align: center; color: #51cf66;'>🏆 {source_team} Wins (+{net_gain_a:.2f})</h4>", unsafe_allow_html=True)
                elif net_gain_b > 0:
                    st.markdown(f"<h4 style='text-align: center; color: #4dabf7;'>🏆 {dest_team} Wins (+{net_gain_b:.2f})</h4>", unsafe_allow_html=True)
                else:
                    st.markdown("<h4 style='text-align: center; color: #adb5bd;'>⚖️ Fair Trade</h4>", unsafe_allow_html=True)
            with res_col3:
                st.metric(label=f"Value Given Up by {dest_team}", value=round(score_b, 2), delta=f"{net_gain_b:+.2f}" if net_gain_b != 0 else "Even")
                
            if not df_a.empty or not df_b.empty:
                st.markdown(f"##### Category Output Deltas ({source_team} Perspective)")
                st.caption(f"*This shows the net statistical gain/loss for **{source_team}** if the trade is executed.*")
                
                delta_cols = ["G", "A", "PPP", "S", "+/-", "Hits", "BkS", "W", "SV%", "GAA", "SO"]
                deltas = {}
                
                has_goalie_a = not df_a[df_a["Position_Type"] == "Goalie"].empty if not df_a.empty else False
                has_goalie_b = not df_b[df_b["Position_Type"] == "Goalie"].empty if not df_b.empty else False
                
                for col in delta_cols:
                    if col in ["SV%", "GAA"] and not (has_goalie_a and has_goalie_b):
                        continue
                    sum_a = pd.to_numeric(df_a[col], errors='coerce').sum() if col in df_a.columns else 0
                    sum_b = pd.to_numeric(df_b[col], errors='coerce').sum() if col in df_b.columns else 0
                    delta = sum_b - sum_a
                    
                    if col in ["SV%", "GAA"]:
                        mean_a = pd.to_numeric(df_a[col], errors='coerce').mean() if (col in df_a.columns and not df_a[col].isnull().all()) else 0
                        mean_b = pd.to_numeric(df_b[col], errors='coerce').mean() if (col in df_b.columns and not df_b[col].isnull().all()) else 0
                        if mean_a == 0 and mean_b == 0: continue
                        delta = mean_b - mean_a
                        
                    if delta != 0:
                        deltas[col] = delta
                            
                if deltas:
                    d_html = "<div style='display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px;'>"
                    for stat, d_val in deltas.items():
                        if stat in ["SV%", "GAA"]:
                            d_str = f"{d_val:+.3f}" if stat == "SV%" else f"{d_val:+.2f}"
                        else:
                            d_str = f"{d_val:+.0f}"
                            
                        if stat == "GAA":
                            c_class = "delta-positive" if d_val < 0 else "delta-negative" if d_val > 0 else "delta-neutral"
                        else:
                            c_class = "delta-positive" if d_val > 0 else "delta-negative" if d_val < 0 else "delta-neutral"
                            
                        d_html += f"<div style='background: #1e2530; padding: 10px 15px; border-radius: 8px; border: 1px solid #2d3748;'><strong>{stat}:</strong> <span class='{c_class}'>{d_str}</span></div>"
                    d_html += "</div>"
                    st.markdown(d_html, unsafe_allow_html=True)

    with tab_ir:
        col_form, col_list = st.columns([1, 1.2])

        with col_form:
            st.subheader("Log an Injury")
            with st.form("injury_form"):
                all_rostered_names = sorted(unified_pool[unified_pool["Fantasy_Team"] != "Waiver Wire / Free Agent"]["Clean_Player"].unique().tolist()) if not unified_pool.empty else []
                if all_rostered_names:
                    inj_player = st.selectbox("Select Rostered Player", options=all_rostered_names)
                else:
                    inj_player = st.text_input("Player Name (Exact)")

                inj_team = st.selectbox("Fantasy Team", all_teams_in_league)
                inj_date = st.date_input("Date Placed on IR", datetime.date.today())
                inj_status = st.selectbox("Designation", ["IR", "IR+", "Out (O)", "Day-to-Day (DTD)", "Season Ending"])
                inj_notes = st.text_input("Injury Notes (e.g. Upper body, week-to-week)")
                submit_injury = st.form_submit_button("Record Injury", use_container_width=True)

                if submit_injury and inj_player:
                    clean_entry = inj_player.replace(" (d)", "").replace(" (f)", "")
                    local_injuries[clean_entry] = {
                        "Fantasy_Team": inj_team,
                        "Date": inj_date.strftime("%Y-%m-%d"),
                        "Status": inj_status,
                        "Notes": inj_notes
                    }
                    save_injuries(local_injuries)
                    st.success(f"Recorded {inj_status} for {inj_player}.")
                    st.rerun()

        with col_list:
            st.subheader("Hospital Ward (Active IR)")
            if local_injuries:
                ir_rows = []
                for p, d in local_injuries.items():
                    row = {"Player": p}
                    row.update(d)
                    ir_rows.append(row)
                ir_df = pd.DataFrame(ir_rows).sort_values(by="Date", ascending=False)
                st.dataframe(ir_df, use_container_width=True, hide_index=True)

                st.markdown("---")
                heal_target = st.selectbox("Activate / Heal Player", options=[""] + list(local_injuries.keys()))
                if st.button("Activate Player (Remove from IR)", use_container_width=True):
                    if heal_target and heal_target in local_injuries:
                        del local_injuries[heal_target]
                        save_injuries(local_injuries)
                        st.success(f"{heal_target} activated and returned to full health.")
                        st.rerun()
            else:
                st.info("No active injuries recorded.")


# ==========================================
# TAB 3: MATH & FORMULA INSPECTOR
# ==========================================
elif app_mode == "📐 Math & Formula Inspector":
    st.title("📐 Math & Formula Inspector")
    st.markdown("Explore the absolute tiered grading scale and Goalie 4-category math model.")

    st.markdown(
        """
        <div class="math-box">
            <h3>1. Skater Absolute Tiered Grading Scale (1.0 to 7.0)</h3>
            <p>Every skater is evaluated equally across the 7 true category slots (Goals, Assists, PPP, Plus/Minus, Shots, Hits, Blocked Shots) using fixed 1-10 level thresholds mapped directly to a 1.0–7.0 scale (Hard Ceiling at 7.0):</p>
            <ul>
                <li><strong>Elite Levels (7 to 10):</strong> Highlighted in bold green.</li>
                <li><strong>Mid Levels (4 to 6):</strong> Neutral / standard performance (uncolored).</li>
                <li><strong>Liability Levels (1 to 3):</strong> Highlighted in soft red.</li>
            </ul>
        </div>

        <div class="math-box">
            <h3>2. Roster Versatility Bonus (Dual Eligibility)</h3>
            <p>Players assigned multiple fantasy positions (e.g., C/LW, LW/RW) receive an artificial <strong>+0.20 point bump</strong> to their Final Rating for each extra position beyond their primary role.</p>
        </div>

        <div class="math-box">
            <h3>3. Goalie 4-Category Tiered Grading Scale (1.0 to 4.0)</h3>
            <p>Goalies are evaluated across their 4 specific categories (Wins, GAA, Save Percentage, Shutouts) using fixed 1-10 level thresholds mapped directly to a 1.0–4.0 scale:</p>
            <ul>
                <li><strong>Wins & Shutouts:</strong> Higher volume yields higher tiers.</li>
                <li><strong>GAA (Goals Against Average):</strong> Evaluated inversely (lower GAA receives elite green tiers).</li>
                <li><strong>Save Percentage (SV%):</strong> Absolute threshold tiers ranging from sub-.880 to .925+.</li>
            </ul>
        </div>

        <div class="math-box">
            <h3>4. Durability & Availability Weighting</h3>
            <p>To ensure part-time grinders or injury-prone players don't artificially outrank iron-man top-liners, a Games Played (GP) factor is applied:</p>
            <p><code>Final Rating = 1.0 + (Rating Out of Max - 1.0) * max(0.3, GP / Baseline)</code></p>
            <p>Players missing significant chunks of the season take a durability discount. Use the '82-Game Pace' toggle in the player pool filter to view raw per-game stat accumulation rates.</p>
        </div>
        
        <div class="math-box">
            <h3>5. Multi-Season 80/50/20 Aggregation</h3>
            <p>When viewing the Multi-Season filter, stats are aggregated via a weighted average:</p>
            <ul>
                <li><strong>Most Recent Season (2025-26):</strong> 80% Weight (0.8)</li>
                <li><strong>Prior Season (2024-25):</strong> 50% Weight (0.5)</li>
                <li><strong>Two Seasons Ago (2023-24):</strong> 20% Weight (0.2)</li>
            </ul>
            <p>The total weighted stats are divided by 1.5 to output a true, single-season 82-game pace equivalent.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# TAB 4: THE WEEKLY GAZETTE
# ==========================================
elif app_mode == "📰 Weekly Gazette / League Journal":
    st.title("📰 The Weekly Gazette & League Journal")
    st.markdown("Your automated dose of league history, bad beats, and waiver wire miracles.")
    
    gazette_options = ["Season to Date", "Last 4 Weeks"] + list(WEEKLY_SCHEDULE.keys())
    
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_week = st.selectbox("Select Matchup Week or Range", gazette_options)
    
    today = datetime.date.today()
    
    if selected_week == "Season to Date":
        start_d = "2026-09-29"
        end_d = today.strftime("%Y-%m-%d")
        playoff_tag = " **(Full Year)**"
    elif selected_week == "Last 4 Weeks":
        start_d = (today - datetime.timedelta(days=28)).strftime("%Y-%m-%d")
        end_d = today.strftime("%Y-%m-%d")
        playoff_tag = " **(Rolling 28 Days)**"
    else:
        week_info = WEEKLY_SCHEDULE[selected_week]
        start_d = week_info["start"]
        end_d = week_info["end"]
        playoff_tag = f" **🏆 {week_info['playoffs']}**" if "playoffs" in week_info else ""
    
    with col_info:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Data Window:** {start_d} to {end_d}{playoff_tag}")
        
    with st.spinner(f"Pulling NHL Box Scores for {selected_week}..."):
        wk_skaters = fetch_streak_data(start_d, end_d, fetch_goalies=False, pace_82=True)
        wk_goalies = fetch_streak_data(start_d, end_d, fetch_goalies=True, pace_82=True)
        
    if wk_skaters.empty and wk_goalies.empty:
        st.warning(f"No NHL game data found between {start_d} and {end_d}. Check back once games have been played.")
    else:
        if not wk_skaters.empty:
            wk_skaters["Fantasy_Team"] = wk_skaters["Clean_Player"].apply(lambda x: drafted_player_map.get(x, "Waiver Wire / Free Agent"))
        if not wk_goalies.empty:
            wk_goalies["Fantasy_Team"] = wk_goalies["Clean_Player"].apply(lambda x: drafted_player_map.get(x, "Waiver Wire / Free Agent"))
        
        st.markdown("---")
        st.subheader("📰 Front Page Headlines")
        
        if not wk_skaters.empty:
            team_scoring = wk_skaters[wk_skaters["Fantasy_Team"] != "Waiver Wire / Free Agent"].groupby("Fantasy_Team")[["G", "A", "Hits", "BkS"]].sum().reset_index()
            if not team_scoring.empty and len(team_scoring) > 1:
                team_scoring["Points"] = team_scoring["G"] + team_scoring["A"]
                juggernaut = team_scoring.sort_values(by="Points", ascending=False).iloc[0]
                basement = team_scoring.sort_values(by="Points", ascending=True).iloc[0]
                
                st.success(f"🔥 **THE JUGGERNAUT:** **{juggernaut['Fantasy_Team']}** chose violence this period, hanging a league-leading {juggernaut['Points']:.0f} points ({juggernaut['G']:.0f}G, {juggernaut['A']:.0f}A) on their opponents. Absolutely ruthless.")
                st.error(f"🧊 **THE ICE COLD:** Thoughts and prayers to **{basement['Fantasy_Team']}**, whose roster collectively forgot how to play hockey, mustering a pitiful {basement['Points']:.0f} total points. Time to hit the waiver wire.")

        st.markdown("---")
        st.subheader("🎭 The League Award Ceremony")
        st.markdown("Celebrating the heroes, the villains, and the guys who just showed up for the free Gatorade. *(Reload the page to see different awards!)*")
        
        available_awards = []
        rostered_skaters = wk_skaters[wk_skaters["Fantasy_Team"] != "Waiver Wire / Free Agent"] if not wk_skaters.empty else pd.DataFrame()
        rostered_goalies = wk_goalies[wk_goalies["Fantasy_Team"] != "Waiver Wire / Free Agent"] if not wk_goalies.empty else pd.DataFrame()
        
        if not rostered_skaters.empty:
            c_pool = rostered_skaters[(rostered_skaters["G"] == 0) & (rostered_skaters["A"] == 0) & (rostered_skaters["Hits"] == 0) & (rostered_skaters["BkS"] == 0)]
            if not c_pool.empty:
                w = c_pool.sort_values(by="GP", ascending=False).iloc[0]
                if w["GP"] >= 2:
                    available_awards.append({
                        "style": "info", "icon": "🏃", "name": "The Cardio King",
                        "player": w["Player"], "team": w["Fantasy_Team"],
                        "desc": f"Suited up for {w['GP']:.0f} games with 0 points, 0 hits, and 0 blocks. Pure aerobic exercise."
                    })
            
            s_pool = rostered_skaters[(rostered_skaters["G"] == 0) & (rostered_skaters["A"] == 0)].copy()
            if not s_pool.empty:
                s_pool["Grit"] = s_pool["Hits"] + s_pool["BkS"]
                w = s_pool.sort_values(by="Grit", ascending=False).iloc[0]
                if w["Grit"] >= 5:
                    available_awards.append({
                        "style": "warning", "icon": "🛡️", "name": "The Human Pincushion",
                        "player": w["Player"], "team": w["Fantasy_Team"],
                        "desc": f"{w['Grit']:.0f} combined hits & blocks with zero offensive production. This man hates his own bones."
                    })

            w = rostered_skaters.sort_values(by="+/-", ascending=True).iloc[0]
            if w["+/-"] <= -3:
                available_awards.append({
                    "style": "error", "icon": "🕳️", "name": "The Plus/Minus Abyss",
                    "player": w["Player"], "team": w["Fantasy_Team"],
                    "desc": f"A catastrophic {w['+/-']:.0f} rating. He was on the ice for so many goals against, he might as well have bought a ticket."
                })

            fa_pool = rostered_skaters[rostered_skaters["Clean_Player"].apply(lambda x: acquisition_map.get(x) == "🔄 Trade/Waiver")]
            if not fa_pool.empty:
                w = fa_pool.sort_values(by="Overall_Rating", ascending=False).iloc[0]
                available_awards.append({
                    "style": "success", "icon": "🎯", "name": "Galaxy-Brain Streamer",
                    "player": w["Player"], "team": w["Fantasy_Team"],
                    "desc": f"Scooped off the scrap heap to deliver {w['G']:.0f}G, {w['A']:.0f}A, and a {w['Overall_Rating']:.2f} Rating. Give this GM a front office job."
                })

            st_pool = rostered_skaters[rostered_skaters["G"] == 0]
            if not st_pool.empty:
                w = st_pool.sort_values(by="S", ascending=False).iloc[0]
                if w["S"] >= 8:
                    available_awards.append({
                        "style": "warning", "icon": "🎯", "name": "The Stormtrooper",
                        "player": w["Player"], "team": w["Fantasy_Team"],
                        "desc": f"Fired {w['S']:.0f} shots on net and hit absolutely nothing. Someone check this man's stick tape."
                    })

            cy_pool = rostered_skaters[(rostered_skaters["G"] > 0) & (rostered_skaters["A"] == 0)]
            if not cy_pool.empty:
                w = cy_pool.sort_values(by="G", ascending=False).iloc[0]
                if w["G"] >= 2:
                    available_awards.append({
                        "style": "info", "icon": "⚾", "name": "The Cy Young Award",
                        "player": w["Player"], "team": w["Fantasy_Team"],
                        "desc": f"An elite {w['G']:.0f}-{w['A']:.0f} stat line. Passing the puck is strictly forbidden for this guy."
                    })

            ph_pool = rostered_skaters[(rostered_skaters["A"] > 0) & (rostered_skaters["G"] == 0)]
            if not ph_pool.empty:
                w = ph_pool.sort_values(by="A", ascending=False).iloc[0]
                if w["A"] >= 3:
                    available_awards.append({
                        "style": "success", "icon": "🤝", "name": "The Philanthropist",
                        "player": w["Player"], "team": w["Fantasy_Team"],
                        "desc": f"Dished out {w['A']:.0f} assists without ever looking at the net. A true man of the people."
                    })

            sg_pool = rostered_skaters[(rostered_skaters["G"] > 0) & (rostered_skaters["G"] == rostered_skaters["S"])]
            if not sg_pool.empty:
                w = sg_pool.sort_values(by="G", ascending=False).iloc[0]
                available_awards.append({
                    "style": "warning", "icon": "👽", "name": "The Sniper Gremlin",
                    "player": w["Player"], "team": w["Fantasy_Team"],
                    "desc": f"Scored {w['G']:.0f} goal(s) on exactly {w['S']:.0f} shot(s). A 100% shooting percentage that defies the laws of physics."
                })

            pp_pool = rostered_skaters[rostered_skaters["PPP"] >= 2]
            if not pp_pool.empty:
                w = pp_pool.sort_values(by="PPP", ascending=False).iloc[0]
                available_awards.append({
                    "style": "info", "icon": "⚡", "name": "The Powerplay Merchant",
                    "player": w["Player"], "team": w["Fantasy_Team"],
                    "desc": f"Racked up {w['PPP']:.0f} powerplay points. Essentially useless at 5-on-5, but a god with the man advantage."
                })

        if not rostered_goalies.empty:
            min_gp = 2 if selected_week in ["Season to Date", "Last 4 Weeks"] else 1
            cg_pool = rostered_goalies[(rostered_goalies["W"] >= 1) & (rostered_goalies["GP"] >= min_gp)]
            if not cg_pool.empty:
                w = cg_pool.sort_values(by="GAA", ascending=False).iloc[0]
                if w["GAA"] >= 3.0:
                    available_awards.append({
                        "style": "error", "icon": "🚨", "name": "Cardiac Arrest Goalie",
                        "player": w["Player"], "team": w["Fantasy_Team"],
                        "desc": f"Secured a Win despite a terrifying {w['GAA']:.2f} GAA. Apologies to this manager's blood pressure."
                    })

            sc_pool = rostered_goalies[rostered_goalies["SV%"] > 0]
            if not sc_pool.empty:
                w = sc_pool.sort_values(by="SV%", ascending=True).iloc[0]
                if w["SV%"] <= 0.860:
                    available_awards.append({
                        "style": "error", "icon": "🧀", "name": "Swiss Cheese Netminder",
                        "player": w["Player"], "team": w["Fantasy_Team"],
                        "desc": f"Stopped beachballs at a dismal {w['SV%']:.3f} clip. Might have been better off using a shooter tutor."
                    })

            bw_pool = rostered_goalies[rostered_goalies["SV%"] > 0]
            if not bw_pool.empty:
                w = bw_pool.sort_values(by="SV%", ascending=False).iloc[0]
                if w["SV%"] >= 0.940 and w["GP"] >= 2:
                    available_awards.append({
                        "style": "success", "icon": "🧱", "name": "The Brick Wall",
                        "player": w["Player"], "team": w["Fantasy_Team"],
                        "desc": f"Put up a ridiculous {w['SV%']:.3f} Save Percentage over {w['GP']:.0f} games. Absolutely unbothered."
                    })

        random.shuffle(available_awards)
        selected_awards = available_awards[:6]
        
        for i in range(0, len(selected_awards), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(selected_awards):
                    aw = selected_awards[i + j]
                    with col:
                        callout = getattr(st, aw["style"])
                        callout(f"{aw['icon']} **{aw['name']}**\n\n**{aw['player']}**\n\n*{aw['team']}*")
                        st.caption(f"**Reporter's Note:** {aw['desc']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
                    
        # --- TRUE STRENGTH MATRIX ---
        st.markdown("---")
        st.subheader("📊 The 'No Excuses' Matchup Matrix & All-Play Record")
        st.markdown("If every team played a head-to-head matchup against *every other team* during this exact timeframe, who would have actually won? This ranking exposes the lucky frauds and vindicates the unlucky.")
        
        team_totals = []
        for t in all_teams_in_league:
            t_data = {"Team": t, "G": 0, "A": 0, "+/-": 0, "PPP": 0, "S": 0, "Hits": 0, "BkS": 0, "W": 0, "SV%": 0.0, "GAA": 0.0, "SO": 0}
            
            if not wk_skaters.empty:
                t_skaters = wk_skaters[wk_skaters["Fantasy_Team"] == t]
                if not t_skaters.empty:
                    for col in ["G", "A", "+/-", "PPP", "S", "Hits", "BkS"]:
                        t_data[col] = t_skaters[col].sum()
                        
            if not wk_goalies.empty:
                t_goalies = wk_goalies[wk_goalies["Fantasy_Team"] == t]
                if not t_goalies.empty and t_goalies["GP"].sum() > 0:
                    t_data["W"] = t_goalies["W"].sum()
                    t_data["SO"] = t_goalies["SO"].sum()
                    t_data["GAA"] = t_goalies["GAA"].mean()
                    t_data["SV%"] = t_goalies["SV%"].mean() 
                else:
                    t_data["GAA"] = 99.99 
                    t_data["SV%"] = 0.00
                    
            team_totals.append(t_data)
            
        for i, t1 in enumerate(team_totals):
            w, l, tie = 0, 0, 0
            for j, t2 in enumerate(team_totals):
                if i == j: continue
                cats_won, cats_lost = 0, 0
                for cat in ["G", "A", "+/-", "PPP", "S", "Hits", "BkS", "W", "SV%", "SO"]:
                    if t1[cat] > t2[cat]: cats_won += 1
                    elif t1[cat] < t2[cat]: cats_lost += 1
                
                if t1["GAA"] < t2["GAA"]: cats_won += 1
                elif t1["GAA"] > t2["GAA"]: cats_lost += 1
                
                if cats_won > cats_lost: w += 1
                elif cats_won < cats_lost: l += 1
                else: tie += 1
                
            t1["All-Play Rank (W-L-T)"] = f"{w}-{l}-{tie}"
            t1["_Sort_W"] = w 
            
        tt_df = pd.DataFrame(team_totals)
        
        for cat in ["G", "A", "+/-", "PPP", "S", "Hits", "BkS", "W", "SV%", "SO"]:
            tt_df[f"{cat}_Rk"] = tt_df[cat].rank(method='average')
            
        tt_df["GAA_Rk"] = tt_df["GAA"].rank(method='average', ascending=False)
        rank_cols = [c for c in tt_df.columns if c.endswith("_Rk")]
        tt_df["Total_Roto"] = tt_df[rank_cols].sum(axis=1)
        tt_df["Rating (/10)"] = (tt_df["Total_Roto"] / 88.0) * 10.0
        tt_df = tt_df.drop(columns=rank_cols + ["Total_Roto"])
        
        tt_df = tt_df.sort_values(by=["_Sort_W", "Rating (/10)"], ascending=[False, False]).drop(columns=["_Sort_W"])
        cols = ["Team", "Rating (/10)", "All-Play Rank (W-L-T)"] + [c for c in tt_df.columns if c not in ["Team", "Rating (/10)", "All-Play Rank (W-L-T)"]]
        tt_df = tt_df[cols]
        
        st.dataframe(
            tt_df.style.highlight_max(subset=["G", "A", "+/-", "PPP", "S", "Hits", "BkS", "W", "SV%", "SO"], color="#51cf66")
                       .highlight_min(subset=["GAA"], color="#51cf66")
                       .highlight_min(subset=["G", "A", "+/-", "PPP", "S", "Hits", "BkS", "W", "SV%", "SO"], color="#ff6b6b")
                       .highlight_max(subset=["GAA"], color="#ff6b6b")
                       .format({
                           "Rating (/10)": "⭐ {:.1f}",
                           "G": "{:.0f}", "A": "{:.0f}", "+/-": "{:.0f}", "PPP": "{:.0f}", 
                           "S": "{:.0f}", "Hits": "{:.0f}", "BkS": "{:.0f}", "W": "{:.0f}", 
                           "SO": "{:.0f}", "GAA": "{:.2f}", "SV%": "{:.3f}"
                       }),
            use_container_width=True, 
            hide_index=True
        )
        st.caption("*(⭐ Rating / 10 is calculated across all 11 categories. Green indicates league-best; Red indicates league-worst. GAA of 99.99 indicates no goalies were played.)*")


# ==========================================
# TAB 5: THE ORACLE PROJECTIONS
# ==========================================
elif app_mode == "🔮 Oracle Projections (Beat the Val Model)":
    st.title("🔮 The Gazette Oracle: Season Forecasting")
    st.markdown("Comparing our custom Tiered Rating engine against the rest of the league to predict the true end-of-season standings.")

    if unified_pool.empty:
        st.warning("No player data available. Please check your date filters.")
    else:
        st.info("💡 **How it works:** The Gazette Model calculates the 'True Talent' of a team by summing the Overall Tiered Ratings (1.0-7.0 for skaters, 1.0-4.0 for goalies) of every active player on their roster.")
        
        proj_data = []
        for t in all_teams_in_league:
            t_pool = unified_pool[unified_pool["Fantasy_Team"] == t]
            if not t_pool.empty:
                skaters = t_pool[t_pool["Position_Type"] == "Skater"]
                goalies = t_pool[t_pool["Position_Type"] == "Goalie"]
                skater_power = skaters["Overall_Rating"].sum() if not skaters.empty else 0.0
                goalie_power = goalies["Overall_Rating"].sum() if not goalies.empty else 0.0
                
                normalized_goalie_power = goalie_power * 1.75
                total_power = skater_power + normalized_goalie_power
                top_players = t_pool.sort_values(by="Overall_Rating", ascending=False).head(3)["Player"].tolist()
                
                proj_data.append({
                    "Team": t,
                    "Total Roster Power": total_power,
                    "Skater Strength": skater_power,
                    "Goalie Strength": goalie_power,
                    "Roster Size": len(t_pool),
                    "Franchise Cornerstones": ", ".join(top_players)
                })
                
        proj_df = pd.DataFrame(proj_data)
        if not proj_df.empty:
            proj_df = proj_df.sort_values(by="Total Roster Power", ascending=False).reset_index(drop=True)
            proj_df.index = proj_df.index + 1
            
            assessments = []
            for rank in proj_df.index:
                if rank == 1:
                    assessments.append("🏆 Championship Favorite (Elite Underlying Talent)")
                elif rank in [2, 3]:
                    assessments.append("⚔️ Heavyweight Contender (Built for the Playoffs)")
                elif rank in [4, 5]:
                    assessments.append("⚖️ Mid-Tier Threat (One trade away from contending)")
                elif rank in [6, 7]:
                    assessments.append("⚠️ Paper Tiger / Rebuild Needed (Statistically weak)")
                else:
                    assessments.append("🪦 The Basement (Mathematically eliminated by the Oracle)")
            
            proj_df.insert(2, "Gazette Model Assessment", assessments)
            
            st.markdown("### 📈 Projected End-of-Season Standings (True Talent)")
            st.dataframe(
                proj_df.style.background_gradient(subset=["Total Roster Power", "Skater Strength", "Goalie Strength"], cmap="viridis")
                             .format({
                                 "Total Roster Power": "{:.2f}",
                                 "Skater Strength": "{:.2f}",
                                 "Goalie Strength": "{:.2f}"
                             }),
                use_container_width=True
            )
            
            st.markdown("---")
            st.subheader("🆚 Model vs. Model Breakdown")
            
            top_team = proj_df.iloc[0]
            bottom_team = proj_df.iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**The Gazette's Chosen Champion:** {top_team['Team']}")
                st.write(f"**Total Power Score:** {top_team['Total Roster Power']:.2f}")
                st.write(f"**Led By:** {top_team['Franchise Cornerstones']}")
                st.caption("This team possesses the highest aggregate multi-category potential in the league.")
                
            with col2:
                st.error(f"**The Gazette's Rebuild Candidate:** {bottom_team['Team']}")
                st.write(f"**Total Power Score:** {bottom_team['Total Roster Power']:.2f}")
                st.write(f"**Led By:** {bottom_team['Franchise Cornerstones']}")
                st.caption("Regardless of their current real-life standing, this roster is mathematically hollow.")