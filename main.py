import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Chance Analyzer Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# Fixed Patterns
# ==========================================
FIXED_COMBOS_TXT = """
A A A A

A
A
A
A

A
 A
  A
   A

A
 A
 A S
   A

A A S A
 A

A A
A A

A S A
A S A

A S A
S S S
A S A

A S S A
S S S S
S S S S
A S S A
"""

PATTERN_NAMES = {
    0: "1. Row (Horizontal)",
    1: "2. Column (Vertical)",
    2: "3. Diagonal",
    3: "4. Skip Diagonal",
    4: "5. Bridge",
    5: "6. Square (2x2)",
    6: "7. Parallel Gaps",
    7: "8. X-Corners",
    8: "9. Large Corners"
}

# ==========================================
# Logic for Pairs & Signs (+/-)
# ==========================================
PLUS_SET = {"8", "10", "Q", "A"}
MINUS_SET = {"7", "9", "J", "K"}

def get_card_sign(card_val):
    val = str(card_val).strip().upper()
    if val in PLUS_SET: return "+"
    if val in MINUS_SET: return "-"
    return "?"

def analyze_custom_pair(df, suit1, suit2):
    results = []
    draw_col = None
    for c in df.columns:
        if 'הגרלה' in str(c) or 'Draw' in str(c):
            draw_col = c
            break
            
    if suit1 not in df.columns or suit2 not in df.columns:
        return []

    s1_signs = df[suit1].apply(get_card_sign)
    s2_signs = df[suit2].apply(get_card_sign)
    pairs_series = s1_signs + s2_signs 

    target_pairs = ["++", "--", "+-", "-+"]
    
    for p in target_pairs:
        matches = (pairs_series == p)
        if matches.any():
            last_seen_idx = matches.idxmax() 
            draw_num = df.iloc[last_seen_idx][draw_col] if draw_col else f"#{last_seen_idx}"
            c1_val = df.iloc[last_seen_idx][suit1]
            c2_val = df.iloc[last_seen_idx][suit2]
            
            results.append({
                "pair": p,
                "draws_ago": last_seen_idx,
                "draw_num": draw_num,
                "c1_val": c1_val,
                "c2_val": c2_val
            })
    
    results.sort(key=lambda x: x['draws_ago'], reverse=True)
    return results

def analyze_full_4suit_pattern(df, cols):
    if not all(c in df.columns for c in cols):
        return None, 0
        
    sig_series = df[cols[0]].apply(get_card_sign) + \
                 df[cols[1]].apply(get_card_sign) + \
                 df[cols[2]].apply(get_card_sign) + \
                 df[cols[3]].apply(get_card_sign)
                
    import itertools
    combinations = ["".join(p) for p in itertools.product(["+", "-"], repeat=4)]
    
    sleeping_stats = []
    for combo in combinations:
        matches = (sig_series == combo)
        if matches.any():
            last_seen = matches.idxmax()
        else:
            last_seen = len(df) 
            
        sleeping_stats.append((combo, last_seen))
        
    sleeping_stats.sort(key=lambda x: x[1], reverse=True)
    return sleeping_stats[0] 

# ==========================================
# CSS Styling
# ==========================================
st.markdown("""
<style>
    .stApp { direction: ltr; text-align: left; background-color: #0E1117; color: #FAFAFA; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    h1 { margin-bottom: -0.5rem !important; }
    
    div.stButton > button { width: 100%; border-radius: 8px; height: 2.8rem; font-weight: 600; }
    
    .grid-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; background-color: #161B22; padding: 8px; border-radius: 12px; margin-top: 10px; border: 1px solid #30363D; }
    .grid-cell { background-color: #21262D; color: #C9D1D9; padding: 0; text-align: center; border-radius: 6px; position: relative; border: 1px solid #30363D; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: 500; font-family: 'Segoe UI', sans-serif; }
    .missing-circle { background-color: #F0F6FC; color: #0D1117; font-weight: 800; border-radius: 6px; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; box-shadow: inset 0 0 8px rgba(0,0,0,0.2); }
    .frame-box { position: absolute; top: 0; left: 0; right: 0; bottom: 0; border-style: solid; border-color: transparent; pointer-events: none; border-radius: 6px; }
    .grid-header { text-align: center; padding-bottom: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .suit-icon { font-size: 22px; line-height: 1; margin-bottom: 2px; }
    .suit-name { font-size: 10px; color: #8B949E; font-weight: bold; text-transform: uppercase; }
    .shape-preview-wrapper { background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px; padding: 10px; display: flex; justify-content: center; align-items: center; height: 100%; }
    
    .pm-plus { color: #238636; font-weight: 900; font-size: 18px; } 
    .pm-minus { color: #DA3633; font-weight: 900; font-size: 18px; } 
    .pm-cell { text-align: center; border-bottom: 1px solid #30363D; padding: 8px; }
    
    .pair-card { background-color: #161B22; border: 1px solid #30363D; border-radius: 10px; padding: 15px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
    .pair-title { font-size: 18px; font-weight: bold; color: #E6EDF3; display: flex; align-items: center; gap: 10px; }
    .pair-badge { padding: 2px 8px; border-radius: 4px; font-size: 14px; font-weight: bold; }
    .badge-plus { background-color: #238636; color: white; }
    .badge-minus { background-color: #DA3633; color: white; }
    .pair-stat { text-align: right; font-size: 13px; color: #8B949E; }
    .pair-val { font-size: 20px; font-weight: bold; color: #58A6FF; }
    .sleeper-alert { border: 1px solid #D29922; }
    
    [data-testid="stDataFrame"] th { text-align: left !important; }
    [data-testid="stDataFrame"] td { text-align: left !important; }
</style>
""", unsafe_allow_html=True)

# --- Logic Functions ---

@st.cache_data
def load_data_robust(uploaded_file):
    """Robust CSV loader that handles Pais file structure and Hebrew encoding."""
    if uploaded_file is None: return None, "No file"
    
    def rename_cols(df):
        # Includes the "מלתן" typo from the original Pais files!
        hebrew_map = {'תלתן': 'Clubs', 'מלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
        new_cols = {}
        for col in df.columns:
            for heb, eng in hebrew_map.items():
                if heb in col: # Check if the word is *inside* the column name
                    new_cols[col] = eng
        df.rename(columns=new_cols, inplace=True)
        return df

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        df = rename_cols(df)
        return df, "ok"
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp1255')
            df = rename_cols(df)
            return df, "ok"
        except Exception as e2:
            return None, f"שגיאה בקריאת הקובץ: לא ניתן לפענח את הקידוד."

def parse_shapes_strict(text):
    shapes = []
    text = text.replace('\r\n', '\n')
    blocks = text.split('\n\n')
    for block in blocks:
        if not block.strip(): continue
        lines = [l for l in block.split('\n')]
        coords = []
        for r, line in enumerate(lines):
            c_idx = 0; i = 0
            while i < len(line):
                char = line[i]
                if char == 'A': coords.append((r, c_idx)); c_idx += 1
                elif char == 'S': c_idx += 1 
                elif char == ' ':
                    prev = line[i-1] if i > 0 else None
                    next_c = line[i+1] if i < len(line)-1 else None
                    if not (prev in ['A', 'S'] and next_c in ['A', 'S']): c_idx += 1
                i += 1
        if not coords: continue
        min_c = min(c for r, c in coords)
        coords = [(r, c - min_c) for r, c in coords]
        shapes.append(coords)
    return shapes

def generate_variations_strict(shape_idx, base_shape):
    variations = set()
    if shape_idx in [0, 1]: 
        variations.add(tuple(sorted(base_shape))) 
    elif shape_idx == 2:
        variations.add(tuple(sorted(base_shape))) 
        max_c = max(c for r,c in base_shape)
        mirror = [(r, max_c-c) for r,c in base_shape]
        variations.add(tuple(sorted(mirror)))
    elif shape_idx == 3:
        # אופקי בלבד: הצורה המקורית + מראה אופקית (ימינה-שמאלה) ללא סיבובים
        variations.add(tuple(sorted(base_shape)))
        max_c = max(c for r,c in base_shape)
        variations.add(tuple(sorted([(r, max_c - c) for r, c in base_shape])))
    elif shape_idx == 4:
        base = [(0,0), (0,1), (0,3), (1,1)]
        variations.add(tuple(sorted(base)))
        max_r = max(r for r,c in base)
        flipped = sorted([(max_r - r, c) for r, c in base])
        variations.add(tuple(flipped))
        for v in list(variations):
            w = max(c for r,c in v)
            mirror = [(r, w-c) for r,c in v]
            variations.add(tuple(sorted(mirror)))
    else:
        variations.add(tuple(sorted(base_shape)))
        w = max(c for r,c in base_shape)
        mirror_h = sorted([(r, w - c) for r, c in base_shape])
        variations.add(tuple(mirror_h))
        max_r = max(r for r,c in base_shape)
        flip_v = sorted([(max_r - r, c) for r, c in base_shape])
        variations.add(tuple(flip_v))
        flip_hv = sorted([(max_r - r, w - c) for r, c in base_shape])
        variations.add(tuple(flip_hv))
    return [list(v) for v in variations]

def draw_preview_html(shape_coords):
    if not shape_coords: return ""
    min_r = min(r for r,c in shape_coords); min_c = min(c for r,c in shape_coords)
    norm = [(r-min_r, c-min_c) for r,c in shape_coords]
    max_r = max(r for r, c in norm) + 1; max_c = max(c for r, c in norm) + 1
    
    grid_html = f'<div style="display:grid; grid-template-columns: repeat({max_c}, 10px); gap: 3px;">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#58A6FF" if (r, c) in norm else "#21262D"
            border = "1px solid #30363D" if (r, c) not in norm else "1px solid #79C0FF"
            grid_html += f'<div style="width:10px; height:10px; border-radius:2px; background-color:{bg}; border:{border};"></div>'
    grid_html += '</div>'
    return f'<div class="shape-preview-wrapper">{grid_html}</div>'

def create_sleeping_html_table(data_dict, required_cols):
    meta = {
        'Clubs': {'icon': '♣', 'color': '#E1E4E8'},
        'Diamonds': {'icon': '♦', 'color': '#FF4B4B'},
        'Hearts': {'icon': '♥', 'color': '#FF4B4B'},
        'Spades': {'icon': '♠', 'color': '#E1E4E8'}
    }
    max_rows = 0
    clean_data = {}
    for col in required_cols:
        clean_data[col] = data_dict.get(col, [])
        if len(clean_data[col]) > max_rows: max_rows = len(clean_data[col])
            
    parts = []
    parts.append('<div style="overflow-x: auto; border: 1px solid #30363D; border-radius: 6px;">')
    parts.append('<table style="width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px;">')
    parts.append('<thead><tr style="background-color: #161B22; border-bottom: 1px solid #30363D;">')
    
    for col in required_cols:
        c_meta = meta.get(col, {'icon': '', 'color': '#fff'})
        header_content = f"""<div style="display: flex; flex-direction: column; align-items: center;"><div style="font-size: 24px; line-height: 1;">{c_meta['icon']}</div><div style="font-size: 11px; text-transform: uppercase;">{col}</div></div>"""
        parts.append(f'<th style="padding: 10px; text-align: center; color: {c_meta["color"]}; font-weight: bold; border-right: 1px solid #30363D;">{header_content}</th>')
    
    parts.append('</tr></thead><tbody>')
    for i in range(max_rows):
        bg_color = "#0D1117" if i % 2 == 0 else "#161B22"
        parts.append(f'<tr style="background-color: {bg_color};">')
        for col in required_cols:
            val = clean_data[col][i] if i < len(clean_data[col]) else ""
            text_color = meta[col]['color'] if val != "" else "transparent"
            parts.append(f'<td style="padding: 8px; text-align: center; border-right: 1px solid #30363D; color: {text_color};">{val}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)

# --- Full Matrix Generator ---
def create_plus_minus_matrix(df, required_cols):
    rows_to_show = 20
    mini_df = df.head(rows_to_show).copy()
    
    parts = []
    parts.append('<div style="overflow-x: auto; border: 1px solid #30363D; border-radius: 6px;">')
    parts.append('<table style="width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px;">')
    
    parts.append('<thead><tr style="background-color: #161B22; border-bottom: 1px solid #30363D;">')
    parts.append('<th style="padding: 10px; text-align: center; color: #8B949E;">#</th>') 
    
    meta = {
        'Clubs': {'icon': '♣', 'color': '#E1E4E8'},
        'Diamonds': {'icon': '♦', 'color': '#FF4B4B'},
        'Hearts': {'icon': '♥', 'color': '#FF4B4B'},
        'Spades': {'icon': '♠', 'color': '#E1E4E8'}
    }
    
    for col in required_cols:
        c_meta = meta.get(col, {'icon': '', 'color': '#fff'})
        parts.append(f'<th style="padding: 10px; text-align: center; color: {c_meta["color"]}; font-size: 18px;">{c_meta["icon"]}</th>')
    parts.append('</tr></thead><tbody>')
    
    for idx, row in mini_df.iterrows():
        bg_color = "#0D1117" if idx % 2 == 0 else "#161B22"
        parts.append(f'<tr style="background-color: {bg_color};">')
        parts.append(f'<td style="padding: 8px; text-align: center; color: #8B949E; font-size: 12px;">{idx}</td>')
        
        for col in required_cols:
            val = row[col]
            sign = get_card_sign(val)
            sign_html = ""
            if sign == "+":
                sign_html = '<span class="pm-plus">+</span>'
            elif sign == "-":
                sign_html = '<span class="pm-minus">-</span>'
            else:
                sign_html = '<span style="color:#555;">?</span>'
            
            parts.append(f'<td class="pm-cell">{sign_html}</td>')
        parts.append('</tr>')
        
    parts.append("</tbody></table></div>")
    return "".join(parts)

def generate_board_html(grid_data, row_limit, cell_styles):
    html = '<div class="grid-container">'
    headers = [
        ('Spades', '♠', '#E1E4E8'),
        ('Hearts', '♥', '#FF4B4B'),
        ('Diamonds', '♦', '#FF4B4B'),
        ('Clubs', '♣', '#E1E4E8')
    ]
    for name, icon, color in headers:
        html += f'<div class="grid-header"><div class="suit-icon" style="color:{color};">{icon}</div><div class="suit-name">{name}</div></div>'
    
    for r in range(min(len(grid_data), row_limit)):
        for c in range(4):
            val = str(grid_data[r, c]); 
            if val == 'nan': val = ''
            
            style_extra = cell_styles.get((r, c), "")
            inner = val
            
            if "MISSING_MARKER" in style_extra:
                inner = f'<div class="missing-circle">{val}</div>'
                style_extra = style_extra.replace("MISSING_MARKER", "")
            
            if style_extra.strip().startswith("cell-"):
                 html += f'<div class="grid-cell {style_extra}">{inner}</div>'
            else:
                 html += f'<div class="grid-cell">{inner}{style_extra}</div>'
                 
    html += '</div>'
    return html

# --- Main App ---

st.title("Chance Analyzer Pro")

# --- Load Logic ---
df = None
base_shapes = parse_shapes_strict(FIXED_COMBOS_TXT)

with st.sidebar:
    st.header("Upload Data")
    manual_file = st.file_uploader("Upload CSV", type=None)

if manual_file:
    df, msg = load_data_robust(manual_file)
    if df is None:
        st.error(msg)

if df is not None:
    required_cols = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"הקובץ נטען אך חסרות העמודות הבאות: {missing}")
        st.stop()

    grid_data = df[required_cols].values
    ROW_LIMIT = 51
    
    # --- Settings ---
    with st.expander("⚙️ Settings & Inputs", expanded=not st.session_state.get('search_done', False)):
        col_conf, col_prev = st.columns([4, 1])
        with col_conf:
            def format_pattern(idx): return PATTERN_NAMES.get(idx, f"Pattern {idx+1}")
            shape_idx = st.selectbox("Search Pattern", range(len(base_shapes)), format_func=format_pattern, label_visibility="collapsed")
        with col_prev:
            st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)
        
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        
        raw_cards = np.unique(grid_data.astype(str))
        clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
        
        c1, c2, c3 = st.columns(3)
        with c1: card1 = st.selectbox("C1", [""] + clean_cards, key="c1", label_visibility="collapsed")
        with c2: card2 = st.selectbox("C2", [""] + clean_cards, key="c2", label_visibility="collapsed")
        with c3: card3 = st.selectbox("C3", [""] + clean_cards, key="c3", label_visibility="collapsed")
        
        selected_cards = [c for c in [card1, card2, card3] if c != ""]
        
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

        b_search, b_reset = st.columns([3, 1])
        with b_search: run_search = st.button("Search", type="primary")
        with b_reset: reset_btn = st.button("Reset")
        
        if reset_btn:
            st.session_state['search_done'] = False
            st.session_state['selected_match'] = None
            st.rerun()

    # --- Search Logic ---
    found_matches = []
    if (run_search or st.session_state.get('search_done', False)) and len(selected_cards) == 3:
        st.session_state['search_done'] = True
        
        variations = generate_variations_strict(shape_idx, base_shapes[shape_idx])
        rows = min(len(grid_data), ROW_LIMIT)
        colors = ['#FF7B72', '#D2A8FF', '#79C0FF', '#7EE787', '#FFA657']
        
        raw_matches = []
        for shape in variations:
            sh_h = max(r for r,c in shape)+1; sh_w = max(c for r,c in shape)+1
            for r in range(rows - sh_h + 1):
                for c in range(4 - sh_w + 1):
                    vals = []; coords = []
                    try:
                        for dr, dc in shape:
                            vals.append(grid_data[r+dr, c+dc])
                            coords.append((r+dr, c+dc))
                    except: continue
                    matched = 0; used = set()
                    for t in selected_cards:
                        for i, v in enumerate(vals):
                            if i not in used and str(v) == t:
                                used.add(i); matched += 1; break
                    if matched == 3:
                        miss_i = [i for i in range(4) if i not in used][0]
                        m_data = {'coords': tuple(sorted(coords)), 'miss_coords': coords[miss_i], 'miss_val': vals[miss_i], 'full_coords_list': coords}
                        if not any(x['coords'] == m_data['coords'] for x in raw_matches):
                            raw_matches.append(m_data)
        
        raw_matches.sort(key=lambda x: x['miss_coords'][0])
        for i, m in enumerate(raw_matches):
            m['id'] = i + 1; m['color'] = colors[i % len(colors)]
            found_matches.append(m)

    # --- TABS ---
    tab_matches, tab_sleep, tab_pairs = st.tabs(["📋 MATCHES", "💤 SLEEPING", "⚖️ PAIRS"])
    
    selected_match_ids = None 
    
    # ------------------ TAB 1: MATCHES ------------------
    with tab_matches:
        if found_matches:
            raw_df = pd.DataFrame([
                {'Missing Card': m['miss_val'], 'Row': m['miss_coords'][0], 'Hidden_ID': m['id']} 
                for m in found_matches
            ])
            
            grouped_df = raw_df.groupby('Missing Card').agg({'Row': lambda x: sorted(list(x)), 'Hidden_ID': list}).reset_index()
            grouped_df['Count'] = grouped_df['Hidden_ID'].apply(len)
            grouped_df = grouped_df.sort_values(by='Count', ascending=False)
            grouped_df['Count'] = grouped_df['Count'].astype(str)
            grouped_df['Row Indexes'] = grouped_df['Row'].apply(lambda x: ", ".join(map(str, x)))
            
            display_df = grouped_df[['Missing Card', 'Count', 'Row Indexes', 'Hidden_ID']]
            num_rows = len(display_df); calc_height = (num_rows + 1) * 35 + 3
            
            event = st.dataframe(display_df.drop(columns=['Hidden_ID']), hide_index=True, use_container_width=True, selection_mode="single-row", on_select="rerun", height=calc_height)
            if len(event.selection['rows']) > 0:
                selected_match_ids = display_df.iloc[event.selection['rows'][0]]['Hidden_ID']
        else:
            if st.session_state.get('search_done', False): st.info("No matches found.")
            
        # Board (Matches Style)
        st.subheader("Game Board")
        cell_styles = {}
        matches_to_show = found_matches
        if selected_match_ids is not None:
            matches_to_show = [m for m in found_matches if m['id'] in selected_match_ids]

        for m in matches_to_show:
            col = m['color']
            for coord in m['full_coords_list']:
                if coord != m['miss_coords']:
                    if coord not in cell_styles: cell_styles[coord] = ""
                    count = cell_styles[coord].count("frame-box"); inset = count * 3
                    cell_styles[coord] += f'<div class="frame-box" style="border-width: 2px; border-color: {col}; top: {inset}px; left: {inset}px; right: {inset}px; bottom: {inset}px;"></div>'
            miss = m['miss_coords']
            if miss not in cell_styles: cell_styles[miss] = ""
            if "MISSING_MARKER" not in cell_styles[miss]: cell_styles[miss] += "MISSING_MARKER"
            
        st.markdown(generate_board_html(grid_data, ROW_LIMIT, cell_styles), unsafe_allow_html=True)

    # ------------------ TAB 2: SLEEPING ------------------
    with tab_sleep:
        sleep_data_lists = {}
        for col_name in required_cols:
            col_idx = required_cols.index(col_name)
            col_data = grid_data[:, col_idx]
            c_unique = np.unique(col_data.astype(str))
            lst = []
            for c in c_unique:
                if str(c).lower() == 'nan': continue
                locs = np.where(col_data == c)[0]
                if len(locs) > 0 and locs[0] > 7: lst.append((c, locs[0]))
            lst.sort(key=lambda x: x[1], reverse=True)
            sleep_data_lists[col_name] = [f"{item[0]} : {item[1]}" for item in lst]

        if any(sleep_data_lists.values()):
            st.markdown(create_sleeping_html_table(sleep_data_lists, required_cols), unsafe_allow_html=True)
        else:
            st.write("No sleeping cards found.")
            
        # Board (Clean)
        st.subheader("Game Board")
        st.markdown(generate_board_html(grid_data, ROW_LIMIT, {}), unsafe_allow_html=True)

    # ------------------ TAB 3: PAIRS (+/-) ------------------
    with tab_pairs:
        # 1. Legend
        st.markdown("""
        <div class="legend-container">
            <div class="legend-box">
                <div class="legend-title txt-plus">PLUS</div>
                <div class="legend-cards">8, 10, Q, A</div>
            </div>
            <div class="legend-box">
                <div class="legend-title txt-minus">MINUS</div>
                <div class="legend-cards">7, 9, J, K</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Controls
        all_suits = [c for c in required_cols if c in df.columns]
        sc1, sc2, sc3 = st.columns([1.5, 1.5, 1])
        with sc1: s_choice1 = st.selectbox("S1", all_suits, index=0, label_visibility="collapsed") # Spade
        with sc2: s_choice2 = st.selectbox("S2", all_suits, index=1, label_visibility="collapsed") # Heart
        with sc3: 
            color_board = st.checkbox("🎨 Color", value=False)
        
        # 3. Calculation & Display
        if s_choice1 == s_choice2:
            st.warning("Select different suits")
        else:
            res = analyze_pair_gap(df, s_choice1, s_choice2)
            best_sleeper = res[0] # The one with biggest gap
            pair_code = best_sleeper['pair'] # e.g. "+-"
            
            s1_sign = "PLUS" if pair_code[0] == "+" else "MINUS"
            s1_cls = "txt-plus" if pair_code[0] == "+" else "txt-minus"
            
            s2_sign = "PLUS" if pair_code[1] == "+" else "MINUS"
            s2_cls = "txt-plus" if pair_code[1] == "+" else "txt-minus"
            
            s_icons = {'Clubs': '♣', 'Diamonds': '♦', 'Hearts': '♥', 'Spades': '♠'}
            ic1 = s_icons.get(s_choice1, "")
            ic2 = s_icons.get(s_choice2, "")

            st.markdown(f"""
            <div class="result-card">
                <div class="result-split">
                    <div class="result-part">
                        <div class="res-suit">{ic1} {s_choice1}</div>
                        <div class="res-val {s1_cls}">{s1_sign}</div>
                    </div>
                    <div class="result-part">
                        <div class="res-suit">{ic2} {s_choice2}</div>
                        <div class="res-val {s2_cls}">{s2_sign}</div>
                    </div>
                </div>
                <div class="sub-stat">HAS NOT APPEARED FOR</div>
                <div class="main-stat">{best_sleeper['ago']}</div>
                <div class="sub-stat">DRAWS</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            for i, other in enumerate(res[1:]):
                with [mc1, mc2, mc3][i]:
                    st.caption(f"{other['pair']} : {other['ago']} ago")
                    
        # 4. Global 4-Suit Sleeper
        st.divider()
        st.markdown("##### 😴 Deepest Sleeper (All 4 Suits)")
        sleeper_combo, sleeper_draws = analyze_full_4suit_pattern(df, required_cols)
        
        combo_html = ""
        suits_icons = [s_icons.get(c, "") for c in required_cols]
        for i, char in enumerate(sleeper_combo):
            cls = "badge-plus" if char == "+" else "badge-minus"
            combo_html += f'<span style="margin-right:5px; font-size:16px;">{suits_icons[i]}</span><span class="pair-badge {cls}" style="margin-right:15px;">{char}</span>'
            
        st.markdown(f"""
        <div style="background:#21262D; padding:10px; border-radius:8px; border:1px solid #30363D; display:flex; justify-content:space-between; align-items:center;">
            <div>{combo_html}</div>
            <div style="text-align:right;">
                <div style="font-size:12px; color:#888;">Not seen for</div>
                <div style="font-size:24px; font-weight:bold; color:#D29922;">{sleeper_draws} <span style="font-size:14px;">draws</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()

        # 5. Full Matrix
        st.markdown("##### 📊 Last 20 Draws Matrix")
        st.markdown(create_plus_minus_matrix(df, required_cols), unsafe_allow_html=True)

        # Board (Color Style if checked)
        st.subheader("Game Board")
        cell_styles = {}
        if color_board:
            for r in range(min(len(grid_data), ROW_LIMIT)):
                for c in range(4):
                    val = str(grid_data[r, c])
                    sign = get_card_sign(val)
                    if sign == "+": cell_styles[(r, c)] = " cell-plus"
                    elif sign == "-": cell_styles[(r, c)] = " cell-minus"
        
        st.markdown(generate_board_html(grid_data, ROW_LIMIT, cell_styles), unsafe_allow_html=True)

else:
    st.info("👋 Upload a CSV file to start.")
