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

# Clean Pattern Names (No Emojis)
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

# --- CSS Styling (Clean & Mobile Optimized) ---
st.markdown("""
<style>
    /* Global Settings */
    .stApp { direction: ltr; text-align: left; background-color: #121212; color: #e0e0e0; }
    
    /* Clean Inputs */
    .stSelectbox, .stMultiSelect, .stButton, div[data-testid="stExpander"], div[data-testid="stSidebar"] { 
        direction: ltr; text-align: left; 
    }
    
    /* Mobile Spacing Optimization */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    /* === The Visual Grid === */
    .grid-container { 
        display: grid; 
        grid-template-columns: repeat(4, 1fr); 
        gap: 3px; 
        background-color: #1e1e1e; 
        padding: 6px; 
        border-radius: 8px; 
        margin-top: 10px; 
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
        height: 36px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
    }
    
    /* Missing Card - High Contrast */
    .missing-circle { 
        background-color: #ffffff; 
        color: #000000; 
        font-weight: 900; 
        border-radius: 4px; /* Slightly rounded square looks better on mobile grids */
        width: 100%; 
        height: 100%; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        box-shadow: inset 0 0 5px rgba(0,0,0,0.5);
    }
    
    /* Frames for existing cards */
    .frame-box { 
        position: absolute; 
        top: 0; left: 0; right: 0; bottom: 0; 
        border-style: solid; 
        border-color: transparent; 
        pointer-events: none; 
        border-radius: 4px;
    }
    
    /* Grid Headers (Suits) */
    .grid-header { 
        text-align: center; 
        padding-bottom: 4px; 
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center;
    }
    
    .suit-icon { font-size: 24px; line-height: 1; margin-bottom: 0px; }
    .suit-name { font-size: 10px; color: #888; font-weight: bold; text-transform: uppercase; }
    
    /* Buttons */
    div.stButton > button { 
        width: 100%; 
        border-radius: 6px; 
        font-weight: 600; 
        height: 3rem;
    }
    
    /* Shape Preview Box */
    .shape-preview-wrapper {
        background-color: #222;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 10px;
    }
    
    /* Fix Input Spacing */
    div[data-testid="column"] { gap: 0.3rem; }
    
    /* Results Table Clean Up */
    .dataframe { font-size: 12px !important; }
    
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
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
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
    
    # Clean grid preview
    grid_html = f'<div style="display:grid; grid-template-columns: repeat({max_c}, 18px); gap: 2px;">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#007acc" if (r, c) in norm else "#333"
            border = "1px solid #555" if (r, c) not in norm else "1px solid #0098ff"
            grid_html += f'<div style="width:18px; height:18px; border-radius:2px; background-color:{bg}; border:{border};"></div>'
    grid_html += '</div>'
    return f'<div class="shape-preview-wrapper">{grid_html}</div>'

# --- Main Interface ---

st.title("Chance Analyzer")

# Sidebar Upload
with st.sidebar:
    st.header("Data Source")
    csv_file = st.file_uploader("Upload CSV File", type=None)

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
    # Collapsible setup to save space after search
    with st.expander("Settings & Inputs", expanded=not st.session_state.get('search_done', False)):
        
        # 1. Pattern Selection
        def format_pattern(idx): return PATTERN_NAMES.get(idx, f"Pattern {idx+1}")
        
        shape_idx = st.selectbox("Select Pattern", range(len(base_shapes)), format_func=format_pattern)
        st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)
        
        # 2. Card Selection
        raw_cards = np.unique(grid_data.astype(str))
        clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
        
        st.write("Select 3 Cards:")
        c1_col, c2_col, c3_col = st.columns(3)
        with c1_col: c1 = st.selectbox("C1", [""] + clean_cards, key="c1", label_visibility="collapsed")
        with c2_col: c2 = st.selectbox("C2", [""] + clean_cards, key="c2", label_visibility="collapsed")
        with c3_col: c3 = st.selectbox("C3", [""] + clean_cards, key="c3", label_visibility="collapsed")
        
        selected_cards = [c for c in [c1, c2, c3] if c != ""]
        
        st.write("")
        
        # 3. Action Buttons
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

    # --- RESULTS TABLE ---
    if found_matches:
        st.success(f"Found {len(found_matches)} matches")
        
        df_res = pd.DataFrame([{'ID': m['id'], 'Missing': m['miss_val'], 'Row': m['miss_coords'][0]} for m in found_matches])
        
        event = st.dataframe(
            df_res, 
            hide_index=True, 
            use_container_width=True, 
            selection_mode="single-row", 
            on_select="rerun",
            height=150
        )
        
        selected_match_id = None
        if len(event.selection['rows']) > 0:
            selected_match_id = df_res.iloc[event.selection['rows'][0]]['ID']
    else:
        selected_match_id = None
        if st.session_state.get('search_done', False):
            st.warning("No matches found")

    # --- VISUAL BOARD ---
    st.write("---")
    
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

    # --- SLEEPING SECTION (Hidden by default) ---
    st.write("")
    with st.expander("Sleeping Cards (>7)", expanded=False):
        sleep_cols = st.columns(4)
        icon_map = {'Clubs': '♣', 'Diamonds': '♦', 'Hearts': '♥', 'Spades': '♠'}
        color_map = {'Clubs': '#bbb', 'Diamonds': '#ff5555', 'Hearts': '#ff5555', 'Spades': '#bbb'}
        
        for i, col_name in enumerate(required_cols):
            with sleep_cols[i]:
                st.markdown(f"<div style='text-align:center; font-size:20px; color:{color_map[col_name]}'>{icon_map[col_name]}</div>", unsafe_allow_html=True)
                col_data = grid_data[:, i]
                c_unique = np.unique(col_data.astype(str))
                lst = []
                for c in c_unique:
                    if str(c).lower() == 'nan': continue
                    locs = np.where(col_data == c)[0]
                    if len(locs) > 0 and locs[0] > 7: lst.append((c, locs[0]))
                lst.sort(key=lambda x: x[1], reverse=True)
                
                if lst:
                    for c, g in lst: 
                        st.markdown(f"<div style='text-align:center; font-size:12px; margin-bottom:2px;'><b>{c}</b>: {g}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='text-align:center; color:#555; font-size:12px;'>-</div>", unsafe_allow_html=True)

else:
    st.info("👆 Tap the sidebar arrow to upload CSV.")
