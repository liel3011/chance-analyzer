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

# --- CSS Styling (Fixed Mobile Layout) ---
st.markdown("""
<style>
    /* 1. Global Reset - LOCK WIDTH */
    .stApp { 
        overflow-x: hidden !important;
        width: 100vw !important;
    }
    
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 3rem;
        padding-left: 2px !important;
        padding-right: 2px !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }

    /* 2. FORCE ELEMENTS TO STAY IN ROW */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        white-space: nowrap !important;
        overflow-x: hidden !important;
        gap: 2px !important;
    }
    
    /* 3. Force columns to fit on screen */
    div[data-testid="column"] {
        flex: 1 1 0px !important;
        min-width: 0px !important;
        width: 0px !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    
    /* 4. Tiny Inputs for Mobile */
    div[data-baseweb="select"] > div {
        font-size: 11px !important;
        min-height: 30px !important;
        height: 30px !important;
        padding: 0 2px !important;
    }
    
    /* 5. Compact Tables */
    .dataframe { font-size: 10px !important; }
    
    /* 6. Sleeping Table HTML */
    .sleeping-table {
        width: 100% !important; 
        table-layout: fixed !important; 
        border-collapse: collapse; 
        color: #ddd;
        font-size: 10px; 
        text-align: center;
    }
    .sleeping-table th { padding: 2px; border-bottom: 1px solid #444; background: #222; }
    .sleeping-table td { padding: 1px; border-right: 1px solid #333; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sleeping-table td:last-child { border-right: none; }
    
    /* 7. Grid Cells */
    .grid-container { 
        display: grid; 
        grid-template-columns: repeat(4, 1fr); 
        gap: 1px; 
        background-color: #1e1e1e; 
        padding: 2px; 
        border-radius: 4px; 
        margin-top: 5px; 
        border: 1px solid #333;
        width: 100% !important;
        box-sizing: border-box;
    }
    .grid-cell { 
        background-color: #2d2d2d; color: #cccccc; padding: 0; text-align: center; 
        border-radius: 2px; font-family: sans-serif; font-size: 12px; 
        border: 1px solid #3a3a3a; height: 30px; 
        display: flex; align-items: center; justify-content: center; 
    }
    
    /* 8. Buttons */
    div.stButton > button { 
        width: 100%; border-radius: 4px; height: 2.2rem; 
        font-weight: bold; padding: 0; margin: 0; font-size: 12px;
    }
    
    /* Hide Labels */
    label[data-testid="stLabel"] { display: none; }
    
    /* Suit Icons in Grid */
    .suit-icon { font-size: 14px; margin:0; line-height:1; }
    
    /* Remove padding from expanders */
    .streamlit-expanderHeader { padding: 5px !important; margin: 0 !important; }
    .streamlit-expanderContent { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# --- Logic ---

@st.cache_data
def load_data_debug(uploaded_file):
    if uploaded_file is None: return None, "No file"
    
    # 1. Excel Check
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
        hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
        df.rename(columns=hebrew_map, inplace=True)
        return df, "ok"
    except Exception as e:
        pass # Not Excel or missing openpyxl

    # 2. CSV Check (Multiple Encodings)
    encodings = ['utf-8', 'cp1255', 'latin1']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            
            # Auto-detect headers based on common names
            cols = [str(c).strip() for c in df.columns]
            df.columns = cols
            
            # Map Hebrew
            hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
            df.rename(columns=hebrew_map, inplace=True)
            
            # Verify we have the 4 suits
            required = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
            if all(r in df.columns for r in required):
                return df, "ok"
        except:
            continue
            
    return None, "Could not read file. Make sure it is CSV/Excel with columns: Clubs, Diamonds, Hearts, Spades (or Hebrew names)."

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
    
    grid_html = f'<div style="display:grid; grid-template-columns: repeat({max_c}, 10px); gap: 1px;">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#007acc" if (r, c) in norm else "#333"
            grid_html += f'<div style="width:10px; height:10px; border-radius:1px; background-color:{bg};"></div>'
    grid_html += '</div>'
    return f'<div style="background:#222; border:1px solid #444; border-radius:4px; padding:4px; display:flex; justify-content:center; align-items:center; height:34px;">{grid_html}</div>'

# --- Main Interface ---

st.title("Chance Analyzer")

# Sidebar
with st.sidebar:
    st.markdown("### Upload")
    csv_file = st.file_uploader("Upload CSV/Excel", type=None)

df = None
base_shapes = parse_shapes_strict(FIXED_COMBOS_TXT)

if csv_file:
    df, msg = load_data_debug(csv_file)
    if df is None: st.error(msg) # Show specific error

if df is not None:
    required_cols = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    grid_data = df[required_cols].values
    ROW_LIMIT = 51
    
    # === 1. SETTINGS ROW ===
    c_pat, c_prev = st.columns([3, 1])
    with c_pat:
        st.markdown("<div style='font-size:10px; color:#888;'>Pattern</div>", unsafe_allow_html=True)
        shape_idx = st.selectbox("Pattern", range(len(base_shapes)), format_func=lambda i: PATTERN_NAMES.get(i, f"Pat {i+1}"), key="p_sel")
    with c_prev:
        st.markdown("<div style='font-size:10px; color:#888;'>Preview</div>", unsafe_allow_html=True)
        st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)
    
    # === 2. CARDS ROW (Forced) ===
    st.markdown("<div style='font-size:10px; color:#888; margin-top:5px;'>Select 3 Cards</div>", unsafe_allow_html=True)
    raw_cards = np.unique(grid_data.astype(str))
    clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
    
    c1, c2, c3 = st.columns(3)
    with c1: s1 = st.selectbox("C1", [""] + clean_cards, key="c1")
    with c2: s2 = st.selectbox("C2", [""] + clean_cards, key="c2")
    with c3: s3 = st.selectbox("C3", [""] + clean_cards, key="c3")
    selected_cards = [c for c in [s1, s2, s3] if c != ""]
    
    # === 3. BUTTONS ROW ===
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

    # === 4. TABLES (Forced Side-by-Side) ===
    st.write("")
    col_res, col_sleep = st.columns(2)
    
    with col_res:
        with st.expander(f"📋 Matches ({len(found_matches)})", expanded=bool(found_matches)):
            if found_matches:
                df_res_display = pd.DataFrame([{'Miss': m['miss_val'], 'Row': m['miss_coords'][0]} for m in found_matches])
                event = st.dataframe(df_res_display, hide_index=True, use_container_width=True, selection_mode="single-row", on_select="rerun", height=200)
                if len(event.selection['rows']) > 0:
                    idx = event.selection['rows'][0]
                    st.session_state['selected_match'] = found_matches[idx]['id']
            else:
                st.session_state['selected_match'] = None
                if st.session_state.get('search_done', False): st.caption("None")

    with col_sleep:
        with st.expander("💤 Sleeping", expanded=False):
            html_table = "<table class='sleeping-table'><thead><tr>"
            icon_map = {'Clubs': '♣', 'Diamonds': '♦', 'Hearts': '♥', 'Spades': '♠'}
            color_map = {'Clubs': '#bbb', 'Diamonds': '#ff5555', 'Hearts': '#ff5555', 'Spades': '#bbb'}
            
            for col_name in required_cols:
                header_html = f"<div style='color:{color_map[col_name]}; font-size:14px;'>{icon_map[col_name]}</div><div style='font-size:8px; color:#888;'>{col_name[:3]}</div>"
                html_table += f"<th>{header_html}</th>"
            html_table += "</tr></thead><tbody>"
            
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
            
            for r in range(max_len):
                html_table += "<tr>"
                for c in range(4):
                    if r < len(cols_data[c]):
                        val, gap = cols_data[c][r]
                        html_table += f"<td><b>{val}</b>:{gap}</td>"
                    else:
                        html_table += "<td></td>"
                html_table += "</tr>"
            html_table += "</tbody></table>"
            st.markdown(html_table, unsafe_allow_html=True)

    # === 5. VISUAL BOARD ===
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
