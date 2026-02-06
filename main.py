import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Chance Analyzer",
    layout="centered",  # Changed to centered for better mobile focus
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

PATTERN_NAMES = {
    0: "1. Row",
    1: "2. Column",
    2: "3. Diagonal",
    3: "4. ZigZag",
    4: "5. Bridge",
    5: "6. Square",
    6: "7. Parallel",
    7: "8. X-Corners",
    8: "9. Big Corners"
}

# ==========================================

# --- CSS Styling (Clean & Stable) ---
st.markdown("""
<style>
    /* Global Settings */
    .stApp { 
        background-color: #121212; 
        color: #e0e0e0;
    }
    
    /* Buttons - Full Width & Big */
    div.stButton > button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3rem; 
        font-weight: bold; 
        font-size: 16px;
    }
    
    /* Selectboxes - Clean */
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        min-height: 45px; /* Easy to tap */
    }
    
    /* Visual Grid */
    .grid-container { 
        display: grid; 
        grid-template-columns: repeat(4, 1fr); 
        gap: 3px; 
        background-color: #1e1e1e; 
        padding: 5px; 
        border-radius: 8px; 
        margin-top: 15px; 
        border: 1px solid #333;
    }
    .grid-cell { 
        background-color: #2d2d2d; 
        color: #cccccc; 
        padding: 0; 
        text-align: center; 
        border-radius: 4px; 
        font-family: sans-serif; 
        font-size: 14px; 
        border: 1px solid #3a3a3a; 
        height: 40px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
    }
    .missing-circle { 
        background-color: #ffffff; 
        color: #000000; 
        font-weight: 900; 
        border-radius: 50%; 
        width: 28px; 
        height: 28px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        box-shadow: 0 0 5px rgba(255,255,255,0.5);
    }
    .frame-box { 
        position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
        border-style: solid; border-color: transparent; pointer-events: none; border-radius: 4px;
    }
    .grid-header { 
        text-align: center; padding-bottom: 5px; 
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .suit-icon { font-size: 24px; margin:0; line-height:1; }
    
    /* Sleeping Table */
    .sleeping-table {
        width: 100%; border-collapse: collapse; color: #ddd;
        font-size: 14px; text-align: center;
    }
    .sleeping-table th { padding: 8px; border-bottom: 2px solid #444; background: #222; }
    .sleeping-table td { padding: 6px; border-bottom: 1px solid #333; }
    
    /* Expanders */
    div[data-testid="stExpander"] { 
        background-color: #1a1a1a; 
        border-radius: 8px; 
        margin-bottom: 10px;
    }
    
    /* Preview Centering */
    .shape-preview-wrapper {
        display: flex; justify-content: center; padding: 10px; background: #222; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Logic ---

@st.cache_data
def load_data_robust(uploaded_file):
    if uploaded_file is None: return None, "No file"
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        if 'תלתן' in df.columns:
            hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
            df.rename(columns=hebrew_map, inplace=True)
        return df, "ok"
    except:
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            if 'תלתן' in df.columns:
                hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
                df.rename(columns=hebrew_map, inplace=True)
            return df, "ok"
        except:
            return None, "Error: Please upload a valid CSV or Excel file."

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
    if shape_idx in [0, 1]: variations.add(tuple(sorted(base_shape)))
    elif shape_idx == 2:
        variations.add(tuple(sorted(base_shape)))
        max_c = max(c for r,c in base_shape)
        variations.add(tuple(sorted([(r, max_c-c) for r,c in base_shape])))
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
            variations.add(tuple(sorted([(r, w-c) for r,c in v])))
    else:
        variations.add(tuple(sorted(base_shape)))
        w = max(c for r,c in base_shape)
        max_r = max(r for r,c in base_shape)
        variations.add(tuple(sorted([(r, w - c) for r, c in base_shape])))
        variations.add(tuple(sorted([(max_r - r, c) for r, c in base_shape])))
        variations.add(tuple(sorted([(max_r - r, w - c) for r, c in base_shape])))
    return [list(v) for v in variations]

def draw_preview_html(shape_coords):
    if not shape_coords: return ""
    min_r = min(r for r,c in shape_coords); min_c = min(c for r,c in shape_coords)
    norm = [(r-min_r, c-min_c) for r,c in shape_coords]
    max_r = max(r for r, c in norm) + 1; max_c = max(c for r, c in norm) + 1
    
    grid_html = f'<div style="display:grid; grid-template-columns: repeat({max_c}, 15px); gap: 2px;">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#007acc" if (r, c) in norm else "#333"
            grid_html += f'<div style="width:15px; height:15px; border-radius:2px; background-color:{bg};"></div>'
    grid_html += '</div>'
    return f'<div class="shape-preview-wrapper">{grid_html}</div>'

# --- Main Interface ---

st.title("Chance Analyzer")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    csv_file = st.file_uploader("Upload CSV/Excel", type=None)

df = None
base_shapes = parse_shapes_strict(FIXED_COMBOS_TXT)

if csv_file:
    df, msg = load_data_robust(csv_file)
    if df is None: st.error(msg)

if df is not None:
    required_cols = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    if not all(c in df.columns for c in required_cols):
        if df.shape[1] >= 4:
            df = df.iloc[:, :4]
            df.columns = required_cols
        else:
            st.error("Invalid columns")
            st.stop()

    grid_data = df[required_cols].values
    ROW_LIMIT = 51
    
    # --- SETUP (Stacked Vertically) ---
    with st.expander("⚙️ Settings", expanded=not st.session_state.get('search_done', False)):
        
        # Pattern
        shape_idx = st.selectbox("Select Pattern", range(len(base_shapes)), format_func=lambda i: PATTERN_NAMES.get(i, f"Pat {i+1}"))
        st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)
        
        st.write("---")
        
        # Cards (Standard Stack)
        st.write("Select 3 Cards:")
        raw_cards = np.unique(grid_data.astype(str))
        clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
        
        c1 = st.selectbox("Card 1", [""] + clean_cards, key="c1")
        c2 = st.selectbox("Card 2", [""] + clean_cards, key="c2")
        c3 = st.selectbox("Card 3", [""] + clean_cards, key="c3")
        
        selected_cards = [c for c in [c1, c2, c3] if c != ""]
        
        st.write("")
        col_b1, col_b2 = st.columns(2)
        with col_b1: run_search = st.button("SEARCH", type="primary")
        with col_b2: reset_btn = st.button("RESET")
        
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

    # --- RESULTS (Stacked) ---
    st.write("")
    
    # 1. Matches
    with st.expander(f"📋 Matches ({len(found_matches)})", expanded=bool(found_matches)):
        if found_matches:
            df_res_display = pd.DataFrame([{'Missing': m['miss_val'], 'Row': m['miss_coords'][0]} for m in found_matches])
            event = st.dataframe(df_res_display, hide_index=True, use_container_width=True, selection_mode="single-row", on_select="rerun", height=200)
            if len(event.selection['rows']) > 0:
                idx = event.selection['rows'][0]
                st.session_state['selected_match'] = found_matches[idx]['id']
        else:
            st.session_state['selected_match'] = None
            if st.session_state.get('search_done', False): st.info("No matches found")

    # 2. Sleeping Cards
    with st.expander("💤 Sleeping Cards (>7)", expanded=False):
        # Prepare Data for Table
        icon_map = {'Clubs': '♣', 'Diamonds': '♦', 'Hearts': '♥', 'Spades': '♠'}
        
        cols_data = []
        max_len = 0
        for i in range(4):
            col_data = grid_data[:, i]
            c_unique = np.unique(col_data.astype(str))
            items = []
            for c in c_unique:
                if str(c).lower() == 'nan': continue
                locs = np.where(col_data == c)[0]
                if len(locs) > 0 and locs[0] > 7: items.append((c, locs[0]))
            items.sort(key=lambda x: x[1], reverse=True)
            cols_data.append(items)
            if len(items) > max_len: max_len = len(items)
            
        html_table = "<table class='sleeping-table'><thead><tr>"
        for col_name in required_cols:
            html_table += f"<th>{icon_map[col_name]} {col_name}</th>"
        html_table += "</tr></thead><tbody>"
        
        for r in range(max_len):
            html_table += "<tr>"
            for c in range(4):
                if r < len(cols_data[c]):
                    val, gap = cols_data[c][r]
                    html_table += f"<td><b>{val}</b>: {gap}</td>"
                else:
                    html_table += "<td></td>"
            html_table += "</tr>"
        html_table += "</tbody></table>"
        st.markdown(html_table, unsafe_allow_html=True)

    # --- VISUAL BOARD ---
    st.write("---")
    st.markdown("##### 📊 Game Board")
    
    matches_to_show = found_matches
    if st.session_state.get('selected_match'):
        matches_to_show = [m for m in found_matches if m['id'] == st.session_state['selected_match']]

    cell_styles = {}
    for m in matches_to_show:
        col = m['color']
        for coord in m['full_coords_list']:
            if coord != m['miss_coords']:
                if coord not in cell_styles: cell_styles[coord] = ""
                count = cell_styles[coord].count("frame-box")
                inset = count * 3
                cell_styles[coord] += f'<div class="frame-box" style="border-width: 2px; border-color: {col}; top: {inset}px; left: {inset}px; right: {inset}px; bottom: {inset}px;"></div>'
        miss = m['miss_coords']
        if miss not in cell_styles: cell_styles[miss] = ""
        cell_styles[miss] += "MISSING_MARKER"

    html = '<div class="grid-container">'
    for name, icon, color in [('Clubs', '♣', '#e0e0e0'), ('Diamonds', '♦', '#ff4d4d'), ('Hearts', '♥', '#ff4d4d'), ('Spades', '♠', '#e0e0e0')]:
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
