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

# English Pattern Names
PATTERN_NAMES = {
    0: "➖ Row (Horizontal)",
    1: "Ӏ Column (Vertical)",
    2: "⸁ Diagonal",
    3: "⚡ ZigZag",
    4: "🌉 Bridge",
    5: "🔳 Square (2x2)",
    6: "∿ Parallel Gaps (2x3)",
    7: "✖ X-Corners (3x3)",
    8: "⚓ Large Corners (4x4)"
}

# ==========================================

# --- CSS Styling (English / LTR / Mobile) ---
st.markdown("""
<style>
    /* General Settings - LTR */
    .stApp { direction: ltr; text-align: left; background-color: #121212; color: #f0f0f0; }
    
    /* Align elements to left */
    .stSelectbox, .stMultiSelect, .stButton, div[data-testid="stExpander"], div[data-testid="stSidebar"] { 
        direction: ltr; 
        text-align: left; 
    }
    
    /* === Center Tables === */
    .dataframe { 
        text-align: center !important; 
        margin-left: auto; 
        margin-right: auto;
        width: 100%;
    }
    th { text-align: center !important; }
    td { text-align: center !important; }
    
    /* Headers */
    h3 { text-align: center; margin-bottom: 15px; font-size: 18px; color: #bbb; }
    
    /* === Visual Grid === */
    .grid-container { 
        display: grid; 
        grid-template-columns: repeat(4, 1fr); 
        gap: 3px; 
        background-color: #181818; 
        padding: 8px; 
        border-radius: 12px; 
        margin-top: 5px; 
    }
    
    .grid-cell { 
        background-color: #2b2b2b; 
        color: #ddd; 
        padding: 0; 
        text-align: center; 
        border-radius: 6px; 
        font-family: 'Roboto', sans-serif; 
        font-size: 14px; 
        position: relative; 
        border: 1px solid #3a3a3a; 
        height: 38px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
    }
    
    /* Missing Card Style */
    .missing-circle { 
        background-color: #ffffff; 
        color: #000000; 
        font-weight: 800; 
        border-radius: 50%; 
        width: 30px; 
        height: 30px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.5); 
        z-index: 10;
    }
    
    /* Frames */
    .frame-box { 
        position: absolute; 
        top: 0; left: 0; right: 0; bottom: 0; 
        border-style: solid; 
        border-color: transparent; 
        pointer-events: none; 
        border-radius: 6px;
    }
    
    /* Grid Headers */
    .grid-header { 
        text-align: center; 
        color: #888; 
        font-size: 11px; 
        padding-bottom: 4px; 
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
    }
    
    .suit-icon { font-size: 22px; line-height: 1; margin-bottom: 2px; }
    
    /* Buttons */
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; }
    
    /* Preview Container */
    .shape-preview-container { 
        display: grid; 
        gap: 2px; 
        background-color: #333; 
        padding: 5px; 
        border-radius: 4px; 
        margin: 0 auto; 
        width: fit-content;
    }
    
    /* Spacing Fix */
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
        # Map Hebrew headers to English
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
    
    # 1. Row
    if shape_idx == 0: 
        variations.add(tuple(sorted(base_shape))) 
    
    # 2. Column
    elif shape_idx == 1: 
        variations.add(tuple(sorted(base_shape)))
    
    # 3. Diagonal
    elif shape_idx == 2:
        variations.add(tuple(sorted(base_shape))) 
        max_c = max(c for r,c in base_shape)
        mirror = [(r, max_c-c) for r,c in base_shape]
        variations.add(tuple(sorted(mirror)))
    
    # 4. ZigZag
    elif shape_idx == 3:
        variations.add(tuple(sorted([(0,0), (1,1), (2,2), (1,2)])))
        variations.add(tuple(sorted([(0,0), (1,1), (2,2), (1,0)])))
        variations.add(tuple(sorted([(0,2), (1,1), (2,0), (1,2)])))
        variations.add(tuple(sorted([(0,2), (1,1), (2,0), (1,0)])))
    
    # 5. Bridge (Fixed: Card under Card)
    elif shape_idx == 4:
        # Base: Row 0 has 3 cards, Row 1 has 1 card at index 1
        base = [(0,0), (0,1), (0,3), (1,1)]
        variations.add(tuple(sorted(base)))
        
        # Vertical Flip (Legs up)
        max_r = max(r for r,c in base)
        flipped = sorted([(max_r - r, c) for r, c in base])
        variations.add(tuple(flipped))
        
        # Mirrors
        for v in list(variations):
            w = max(c for r,c in v)
            mirror = [(r, w-c) for r,c in v]
            variations.add(tuple(sorted(mirror)))
            
    # Others (6+): Row based (Mirrors + Flips, No Rotation)
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
    
    grid_html = f'<div class="shape-preview-container" style="grid-template-columns: repeat({max_c}, 15px);">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#007acc" if (r, c) in norm else "#444"
            grid_html += f'<div style="width:15px; height:15px; border-radius:2px; background-color:{bg};"></div>'
    grid_html += '</div>'
    return grid_html

# --- Main Interface ---

st.title("📱 Chance Analyzer")

with st.sidebar:
    st.header("Upload")
    # type=None allows picking any file on mobile
    csv_file = st.file_uploader("Upload CSV", type=None, key="sidebar_uploader")

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
    
    # === Layout ===
    col_right, col_left = st.columns([1.2, 2])
    
    # --- RIGHT COLUMN: Controls ---
    with col_right:
        st.markdown("### ⚙️ Controls")
        
        # 1. Pattern
        def format_pattern_name(idx):
            return PATTERN_NAMES.get(idx, f"Pattern {idx+1}")

        c_pat, c_prev = st.columns([3, 1])
        with c_pat:
            shape_idx = st.selectbox("Pattern:", range(len(base_shapes)), format_func=format_pattern_name)
        with c_prev:
            st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)
        
        # 2. Cards
        raw_cards = np.unique(grid_data.astype(str))
        clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
        
        st.write("Select 3 Cards:")
        sc1, sc2, sc3 = st.columns(3)
        with sc1: c1 = st.selectbox("1", [""] + clean_cards, key="c1", label_visibility="collapsed")
        with sc2: c2 = st.selectbox("2", [""] + clean_cards, key="c2", label_visibility="collapsed")
        with sc3: c3 = st.selectbox("3", [""] + clean_cards, key="c3", label_visibility="collapsed")
        selected_cards = [c for c in [c1, c2, c3] if c != ""]
        
        st.write("")
        
        # 3. Buttons
        b1, b2 = st.columns([2, 1])
        with b1: run_search = st.button("🔍 SEARCH", type="primary", use_container_width=True)
        with b2: reset_btn = st.button("RESET", use_container_width=True)
        
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

        # --- TABLES ---
        st.divider()
        
        # Results Table
        if found_matches:
            st.markdown(f"### 📋 Matches ({len(found_matches)})")
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

        # Sleeping Table
        st.divider()
        st.markdown("### 💤 Sleeping (>7)")
        
        sleep_cols = st.columns(4)
        icon_map = {'Clubs': '♣', 'Diamonds': '♦', 'Hearts': '♥', 'Spades': '♠'}
        color_map = {'Clubs': '#bbb', 'Diamonds': '#ff5555', 'Hearts': '#ff5555', 'Spades': '#bbb'}
        
        for i, col_name in enumerate(required_cols):
            with sleep_cols[i]:
                st.markdown(f"<div style='text-align:center; font-size:24px; color:{color_map[col_name]}'>{icon_map[col_name]}</div>", unsafe_allow_html=True)
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
                        st.markdown(f"<div style='text-align:center; font-size:13px;'><b>{c}</b>: {g}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='text-align:center; color:#555;'>-</div>", unsafe_allow_html=True)

    # --- LEFT COLUMN: VISUAL GRID ---
    with col_left:
        st.markdown("### 📊 Game Board")
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
            html += f'<div class="grid-header"><div class="suit-icon" style="color:{color};">{icon}</div><div>{name}</div></div>'
        
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
    st.info("👈 Please upload the CSV file in the sidebar")
