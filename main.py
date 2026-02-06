import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Chance Analyzer",
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
 A A
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

# Pattern Names
PATTERN_NAMES = {
    0: "1. Row (Horizontal)",
    1: "2. Column (Vertical)",
    2: "3. Diagonal",
    3: "4. ZigZag",
    4: "5. Bridge",
    5: "6. Square (2x2)",
    6: "7. Parallel Gaps",
    7: "8. X-Corners",
    8: "9. Large Corners"
}

# ==========================================

# --- CSS Styling ---
st.markdown("""
<style>
    /* Global Settings */
    .stApp { direction: ltr; text-align: left; background-color: #0E1117; color: #FAFAFA; }
    
    /* Clean Inputs */
    .stSelectbox, .stMultiSelect, div[data-testid="stExpander"] { 
        direction: ltr; text-align: left; 
    }
    
    /* Buttons */
    div.stButton > button { 
        width: 100%; 
        border-radius: 8px; 
        height: 2.8rem; 
        font-weight: 600;
    }
    
    /* Custom Grid Layout */
    .grid-container { 
        display: grid; 
        grid-template-columns: repeat(4, 1fr); 
        gap: 4px; 
        background-color: #161B22; 
        padding: 8px; 
        border-radius: 12px; 
        margin-top: 10px; 
        border: 1px solid #30363D;
    }
    
    .grid-cell { 
        background-color: #21262D; 
        color: #C9D1D9; 
        padding: 0; 
        text-align: center; 
        border-radius: 6px; 
        font-family: 'Segoe UI', Roboto, sans-serif; 
        font-size: 15px; 
        position: relative; 
        border: 1px solid #30363D; 
        height: 40px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-weight: 500;
    }
    
    /* Missing Card Highlight */
    .missing-circle { 
        background-color: #F0F6FC; 
        color: #0D1117; 
        font-weight: 800; 
        border-radius: 6px; 
        width: 100%; height: 100%; 
        display: flex; align-items: center; justify-content: center; 
        box-shadow: inset 0 0 8px rgba(0,0,0,0.2);
    }
    
    /* Frame Overlays */
    .frame-box { 
        position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
        border-style: solid; border-color: transparent; 
        pointer-events: none; border-radius: 6px;
    }
    
    /* Grid Headers */
    .grid-header { 
        text-align: center; padding-bottom: 6px; 
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .suit-icon { font-size: 22px; line-height: 1; margin-bottom: 2px; }
    .suit-name { font-size: 10px; color: #8B949E; font-weight: bold; text-transform: uppercase; }
    
    /* Shape Preview Box */
    .shape-preview-wrapper {
        background-color: #0D1117;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    
    /* Table Centering Hack */
    div[data-testid="stDataFrame"] table {
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] th {
        text-align: center !important;
    }
    div[data-testid="stDataFrame"] td {
        text-align: center !important;
    }

</style>
""", unsafe_allow_html=True)

# --- Logic Functions ---

@st.cache_data
def load_data_robust(uploaded_file):
    if uploaded_file is None: return None, "No file"
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
        df.rename(columns=hebrew_map, inplace=True)
        return df, "ok"
    except:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp1255')
            hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
            df.rename(columns=hebrew_map, inplace=True)
            return df, "ok"
        except:
            return None, "Error loading file"

def parse_shapes_strict(text):
    shapes = []
    text = text.replace('\r\n', '\n')
    blocks = text.split('\n\n')
    for block in blocks:
        if not block.strip(): continue
        lines = [l for l in block.split('\n')]
        coords = []
        for r, line in enumerate(lines):
            c_idx = 0
            i = 0
            while i < len(line):
                char = line[i]
                if char == 'A':
                    coords.append((r, c_idx)); c_idx += 1
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
    if shape_idx == 0: variations.add(tuple(sorted(base_shape))) 
    elif shape_idx == 1: variations.add(tuple(sorted(base_shape)))
    elif shape_idx == 2:
        variations.add(tuple(sorted(base_shape))) 
        max_c = max(c for r,c in base_shape)
        mirror = [(r, max_c-c) for r,c in base_shape]
        variations.add(tuple(sorted(mirror)))
    elif shape_idx == 3:
        variations.add(tuple(sorted([(0,0), (1,1), (2,2), (1,2)])))
        variations.add(tuple(sorted([(0,0), (1,1), (2,2), (1,0)])))
        variations.add(tuple(sorted([(0,2), (1,1), (2,0), (1,2)])))
        variations.add(tuple(sorted([(0,2), (1,1), (2,0), (1,0)])))
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

# --- Main Interface ---

st.title("Chance Analyzer")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    csv_file = st.file_uploader("Choose a CSV file", type=None)

df = None
base_shapes = parse_shapes_strict(FIXED_COMBOS_TXT)

if csv_file:
    df, msg = load_data_robust(csv_file)
    if df is None: st.error(f"Error: {msg}")

if df is not None:
    required_cols = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    df.columns = df.columns.str.strip()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    grid_data = df[required_cols].values
    ROW_LIMIT = 51
    
    # --- 1. SETTINGS & INPUTS (Expander Returned) ---
    with st.expander("⚙️ Settings & Inputs", expanded=not st.session_state.get('search_done', False)):
        
        # Top Row: Pattern & Preview
        col_conf, col_prev = st.columns([4, 1])
        with col_conf:
            def format_pattern(idx): return PATTERN_NAMES.get(idx, f"Pattern {idx+1}")
            shape_idx = st.selectbox("Search Pattern", range(len(base_shapes)), format_func=format_pattern)
        with col_prev:
            st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)
        
        # Card Selection Row
        raw_cards = np.unique(grid_data.astype(str))
        clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
        
        st.caption("Select 3 Cards:")
        c1, c2, c3 = st.columns(3)
        with c1: card1 = st.selectbox("Card 1", [""] + clean_cards, key="c1")
        with c2: card2 = st.selectbox("Card 2", [""] + clean_cards, key="c2")
        with c3: card3 = st.selectbox("Card 3", [""] + clean_cards, key="c3")
        
        selected_cards = [c for c in [card1, card2, card3] if c != ""]
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

        # Action Buttons
        b_search, b_reset = st.columns([3, 1])
        with b_search: 
            run_search = st.button("🔍 ANALYZE BOARD", type="primary")
        with b_reset: 
            reset_btn = st.button("Reset")
        
        if reset_btn:
            st.session_state['search_done'] = False
            st.session_state['selected_match'] = None
            st.rerun()

    # --- SEARCH LOGIC ---
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

    # --- 2. TABS: RESULTS & SLEEPING ---
    
    tab_matches, tab_sleep = st.tabs(["📋 MATCHES FOUND", "💤 SLEEPING CARDS"])
    
    selected_match_id = None
    
    # --- Tab 1: Matches (Centered, No ID) ---
    with tab_matches:
        if found_matches:
            # Create cleaner DataFrame without ID
            df_res = pd.DataFrame([
                {'Missing Card': m['miss_val'], 'Index': m['miss_coords'][0], 'Hidden_ID': m['id']} 
                for m in found_matches
            ])
            
            # Display Table (Using Pandas Styler for centering)
            # We hide the Hidden_ID from view but keep logical tracking via index if needed
            display_df = df_res.drop(columns=['Hidden_ID'])
            
            # Apply styling to center text
            styled_df = display_df.style.set_properties(**{'text-align': 'center'})\
                                        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])

            event = st.dataframe(
                styled_df, 
                hide_index=True, 
                use_container_width=True, 
                selection_mode="single-row", 
                on_select="rerun",
                height=300
            )
            
            if len(event.selection['rows']) > 0:
                selected_row_idx = event.selection['rows'][0]
                # Retrieve the original ID using the index
                selected_match_id = df_res.iloc[selected_row_idx]['Hidden_ID']
        else:
            if st.session_state.get('search_done', False):
                st.info("No matches found for this pattern.")

    # --- Tab 2: Sleeping Cards (Clean Format) ---
    with tab_sleep:
        sleep_data_dict = {}
        max_len = 0
        
        icon_map = {'Clubs': '♣', 'Diamonds': '♦', 'Hearts': '♥', 'Spades': '♠'}
        
        for col_name in required_cols:
            col_idx = required_cols.index(col_name)
            col_data = grid_data[:, col_idx]
            c_unique = np.unique(col_data.astype(str))
            
            lst = []
            for c in c_unique:
                if str(c).lower() == 'nan': continue
                locs = np.where(col_data == c)[0]
                if len(locs) > 0 and locs[0] > 7:
                    lst.append((c, locs[0]))
            
            lst.sort(key=lambda x: x[1], reverse=True)
            
            # Clean Format: "Value : Index"
            formatted_list = [f"{item[0]} : {item[1]}" for item in lst]
            
            header_key = f"{icon_map[col_name]} {col_name}"
            sleep_data_dict[header_key] = formatted_list
            
            if len(formatted_list) > max_len:
                max_len = len(formatted_list)
        
        # Pad with empty strings
        for k in sleep_data_dict:
            while len(sleep_data_dict[k]) < max_len:
                sleep_data_dict[k].append("")
        
        if sleep_data_dict:
            df_sleep = pd.DataFrame(sleep_data_dict)
            
            # Center the sleeping table as well
            styled_sleep = df_sleep.style.set_properties(**{'text-align': 'center'})\
                                         .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
            
            dynamic_height = (max_len * 35) + 40
            st.dataframe(
                styled_sleep, 
                use_container_width=True, 
                height=dynamic_height
            )
        else:
            st.write("No sleeping cards found.")

    # --- 3. VISUAL BOARD ---
    st.subheader("Game Board")
    
    cell_styles = {}
    
    matches_to_show = found_matches
    if selected_match_id is not None:
        matches_to_show = [m for m in found_matches if m['id'] == selected_match_id]

    for m in matches_to_show:
        col = m['color']
        for coord in m['full_coords_list']:
            if coord != m['miss_coords']:
                if coord not in cell_styles: cell_styles[coord] = ""
                count = cell_styles[coord].count("frame-box"); inset = count * 3
                cell_styles[coord] += f'<div class="frame-box" style="border-width: 2px; border-color: {col}; top: {inset}px; left: {inset}px; right: {inset}px; bottom: {inset}px;"></div>'
        
        miss = m['miss_coords']
        if miss not in cell_styles: cell_styles[miss] = ""
        cell_styles[miss] += "MISSING_MARKER"

    html = '<div class="grid-container">'
    
    headers = [('Clubs', '♣', '#E1E4E8'), ('Diamonds', '♦', '#F97583'), ('Hearts', '♥', '#F97583'), ('Spades', '♠', '#E1E4E8')]
    for name, icon, color in headers:
        html += f'<div class="grid-header"><div class="suit-icon" style="color:{color};">{icon}</div><div class="suit-name">{name}</div></div>'
    
    for r in range(min(len(grid_data), ROW_LIMIT)):
        for c in range(4):
            val = str(grid_data[r, c]); 
            if val == 'nan': val = ''
            content = cell_styles.get((r, c), "")
            inner = val
            if "MISSING_MARKER" in content:
                inner = f'<div class="missing-circle">{val}</div>'
                content = content.replace("MISSING_MARKER", "")
            html += f'<div class="grid-cell">{inner}{content}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

else:
    st.info("👋 Upload a CSV file from the sidebar to start.")
