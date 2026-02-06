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
# Fixed Patterns (Combos)
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

# Clean Pattern Names
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
    .stApp { direction: ltr; text-align: left; background-color: #121212; color: #e0e0e0; }
    
    /* Compact Inputs */
    .stSelectbox, .stMultiSelect, .stButton, div[data-testid="stExpander"], div[data-testid="stSidebar"] { 
        direction: ltr; text-align: left; 
    }
    
    /* Remove padding around main block for mobile */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    /* === Center Tables & Headers === */
    .dataframe { 
        text-align: center !important; 
        margin-left: auto; 
        margin-right: auto;
        width: 100%;
        font-size: 13px !important;
    }
    th { text-align: center !important; }
    td { text-align: center !important; }
    
    /* === The Visual Grid === */
    .grid-container { 
        display: grid; 
        grid-template-columns: repeat(4, 1fr); 
        gap: 2px; 
        background-color: #1e1e1e; 
        padding: 4px; 
        border-radius: 8px; 
        margin-top: 5px; 
        border: 1px solid #333;
    }
    
    .grid-cell { 
        background-color: #2d2d2d; 
        color: #cccccc; 
        padding: 0; 
        text-align: center; 
        border-radius: 4px; 
        font-family: 'Roboto', sans-serif; 
        font-size: 14px; 
        position: relative; 
        border: 1px solid #3a3a3a; 
        height: 35px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
    }
    
    /* Missing Card */
    .missing-circle { 
        background-color: #ffffff; 
        color: #000000; 
        font-weight: 900; 
        border-radius: 4px; 
        width: 100%; height: 100%; 
        display: flex; align-items: center; justify-content: center; 
        box-shadow: inset 0 0 5px rgba(0,0,0,0.5);
    }
    
    /* Frames */
    .frame-box { 
        position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
        border-style: solid; border-color: transparent; 
        pointer-events: none; border-radius: 4px;
    }
    
    /* Grid Headers */
    .grid-header { 
        text-align: center; padding-bottom: 2px; 
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .suit-icon { font-size: 20px; line-height: 1; margin-bottom: 0px; }
    .suit-name { font-size: 9px; color: #888; font-weight: bold; text-transform: uppercase; }
    
    /* Preview Box */
    .shape-preview-wrapper {
        background-color: #222;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 5px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 0px;
    }
    
    /* Expander Styling */
    div[data-testid="stExpander"] {
        border: 1px solid #333;
        border-radius: 6px;
        background-color: #1a1a1a;
    }
    
    /* Buttons */
    div.stButton > button { width: 100%; border-radius: 6px; height: 2.5rem; font-weight: bold; }
    
    /* Input columns spacing */
    div[data-testid="column"] { gap: 0.2rem; }
    
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
            return None, "Error"

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
    
    grid_html = f'<div style="display:grid; grid-template-columns: repeat({max_c}, 12px); gap: 2px;">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#007acc" if (r, c) in norm else "#333"
            border = "1px solid #555" if (r, c) not in norm else "1px solid #0098ff"
            grid_html += f'<div style="width:12px; height:12px; border-radius:1px; background-color:{bg}; border:{border};"></div>'
    grid_html += '</div>'
    return f'<div class="shape-preview-wrapper">{grid_html}</div>'

# --- Main Interface ---

st.title("Chance Analyzer")

# Sidebar
with st.sidebar:
    st.header("Upload")
    csv_file = st.file_uploader("Upload CSV", type=None)

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
    
    # --- SETUP AREA ---
    with st.expander("⚙️ Settings", expanded=not st.session_state.get('search_done', False)):
        # Pattern
        def format_pattern(idx): return PATTERN_NAMES.get(idx, f"Pattern {idx+1}")
        c_pat, c_prev = st.columns([3, 1])
        with c_pat:
            shape_idx = st.selectbox("Pattern", range(len(base_shapes)), format_func=format_pattern, label_visibility="collapsed")
        with c_prev:
            st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)
        
        # Cards (Row of 3)
        raw_cards = np.unique(grid_data.astype(str))
        clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
        
        st.caption("Select 3 Cards:")
        c1_col, c2_col, c3_col = st.columns(3)
        with c1_col: c1 = st.selectbox("C1", [""] + clean_cards, key="c1", label_visibility="collapsed")
        with c2_col: c2 = st.selectbox("C2", [""] + clean_cards, key="c2", label_visibility="collapsed")
        with c3_col: c3 = st.selectbox("C3", [""] + clean_cards, key="c3", label_visibility="collapsed")
        
        selected_cards = [c for c in [c1, c2, c3] if c != ""]
        
        st.write("")
        b1, b2 = st.columns(2)
        with b1: run_search = st.button("SEARCH", type="primary")
        with b2: reset_btn = st.button("RESET")
        
        if reset_btn:
            st.session_state['search_done'] = False
            st.session_state['selected_match'] = None
            st.rerun()

    # --- LOGIC ---
    found_matches = []
    if (run_search or st.session_state.get('search_done', False)) and len(selected_cards) == 3:
        st.session_state['search_done'] = True
        variations = generate_variations_strict(shape_idx, base_shapes[shape_idx])
        rows = min(len(grid_data), ROW_LIMIT)
        colors = ['#00ff99', '#ffcc00', '#ff66cc', '#00ccff', '#ff5050', '#cc99ff', '#ffff00']
        
        raw_matches = []
        for shape in variations:
            sh_h = max(r for r,c in shape)+1; sh_w = max(c for r,c in shape)+1
            for r in range(rows - sh_h + 1):
                for c in range(4 - sh_w + 1):
                    vals = []; coords = []
                    try:
                        for dr, dc in shape:
                            vals.append(grid_data[r+dr, c+dc]); coords.append((r+dr, c+dc))
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

    # --- TABLES (Side by Side) ---
    col_res, col_sleep = st.columns(2)
    
    # 1. Results
    with col_res:
        with st.expander(f"📋 Results ({len(found_matches)})", expanded=bool(found_matches)):
            if found_matches:
                # Show only Missing Card and Row (Hide ID)
                df_res_display = pd.DataFrame([{'Missing': m['miss_val'], 'Row': m['miss_coords'][0]} for m in found_matches])
                
                event = st.dataframe(
                    df_res_display, 
                    hide_index=True, 
                    use_container_width=True, 
                    selection_mode="single-row", 
                    on_select="rerun",
                    height=200
                )
                
                selected_match_id = None
                if len(event.selection['rows']) > 0:
                    # Map back to original ID using index
                    idx = event.selection['rows'][0]
                    selected_match_id = found_matches[idx]['id']
            else:
                selected_match_id = None
                if st.session_state.get('search_done', False):
                    st.caption("No matches found")

    # 2. Sleeping Cards (Table Format)
    with col_sleep:
        with st.expander("💤 Sleeping (>7)", expanded=False):
            # Prepare data for 4-column table
            sleep_data = {'♣': [], '♦': [], '♥': [], '♠': []}
            col_map_idx = {'Clubs': 0, 'Diamonds': 1, 'Hearts': 2, 'Spades': 3}
            icon_headers = ['♣', '♦', '♥', '♠']
            
            # Collect data
            max_len = 0
            for col_name in required_cols:
                c_idx = col_map_idx[col_name]
                col_data = grid_data[:, c_idx]
                c_unique = np.unique(col_data.astype(str))
                
                # Get items > 7
                col_items = []
                for c in c_unique:
                    if str(c).lower() == 'nan': continue
                    locs = np.where(col_data == c)[0]
                    if len(locs) > 0 and locs[0] > 7:
                        col_items.append((c, locs[0]))
                
                # Sort descending by gap
                col_items.sort(key=lambda x: x[1], reverse=True)
                
                # Format strings "Card: Gap"
                formatted_items = [f"{c}: {g}" for c, g in col_items]
                sleep_data[icon_headers[c_idx]] = formatted_items
                if len(formatted_items) > max_len: max_len = len(formatted_items)
            
            # Pad lists to same length for DataFrame
            for k in sleep_data:
                while len(sleep_data[k]) < max_len:
                    sleep_data[k].append("")
            
            df_sleep = pd.DataFrame(sleep_data)
            st.dataframe(df_sleep, hide_index=True, use_container_width=True, height=200)

    # --- VISUAL BOARD ---
    st.markdown("##### 📊 Game Board")
    
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
    headers = [('Clubs', '♣', '#e0e0e0'), ('Diamonds', '♦', '#ff4d4d'), ('Hearts', '♥', '#ff4d4d'), ('Spades', '♠', '#e0e0e0')]
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
    st.info("👆 Tap the sidebar arrow to upload CSV.")
