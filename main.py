import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Chance Analyzer",
    layout="centered",
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

# --- CSS Styling (Clean & Simple) ---
st.markdown("""
<style>
    /* Global */
    .stApp { background-color: #121212; color: #e0e0e0; }
    
    /* Buttons */
    div.stButton > button { 
        width: 100%; border-radius: 8px; height: 3rem; font-weight: bold; font-size: 16px;
    }
    
    /* Input Fields */
    div[data-baseweb="select"] > div { border-radius: 8px; min-height: 45px; }
    
    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 16px; font-weight: bold; width: 100%;
    }
    
    /* Grid Styling */
    .grid-container { 
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; 
        background-color: #1e1e1e; padding: 5px; border-radius: 8px; margin-top: 10px; border: 1px solid #333;
    }
    .grid-cell { 
        background-color: #2d2d2d; color: #cccccc; padding: 0; text-align: center; 
        border-radius: 4px; font-family: sans-serif; font-size: 14px; 
        border: 1px solid #3a3a3a; height: 40px; 
        display: flex; align-items: center; justify-content: center; 
    }
    .missing-circle { 
        background-color: #ffffff; color: #000000; font-weight: 900; 
        border-radius: 50%; width: 28px; height: 28px; 
        display: flex; align-items: center; justify-content: center; 
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
    
    /* Sleeping Column Style */
    .sleep-col {
        text-align: center;
        background-color: #1a1a1a;
        border-radius: 6px;
        padding: 5px;
        margin: 2px;
        min-height: 100px;
    }
    .sleep-header {
        font-weight: bold; border-bottom: 1px solid #444; padding-bottom: 5px; margin-bottom: 5px;
    }
    .sleep-item {
        font-size: 13px; margin-bottom: 2px; color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

# --- Logic ---

@st.cache_data
def load_data_robust(uploaded_file):
    if uploaded_file is None: return None, "No file"
    
    # 1. Try Excel
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
        hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
        df.rename(columns=hebrew_map, inplace=True)
        return df, "ok"
    except:
        pass 

    # 2. Try CSV
    for enc in ['utf-8', 'cp1255', 'latin1']:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            # Auto-clean headers
            df.columns = [str(c).strip() for c in df.columns]
            hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
            df.rename(columns=hebrew_map, inplace=True)
            return df, "ok"
        except:
            continue
            
    return None, "Error: Could not read file."

def parse_shapes(text):
    shapes = []
    text = text.replace('\r\n', '\n')
    blocks = text.split('\n\n')
    for block in blocks:
        if not block.strip(): continue
        lines = block.split('\n')
        coords = []
        for r, line in enumerate(lines):
            c_idx = 0
            i = 0
            while i < len(line):
                char = line[i]
                if char == 'A': coords.append((r, c_idx)); c_idx += 1
                elif char == 'S': c_idx += 1 
                elif char == ' ':
                    if i>0 and i<len(line)-1: c_idx += 1
                i += 1
        if coords:
            min_c = min(c for r, c in coords)
            shapes.append([(r, c - min_c) for r, c in coords])
    return shapes

def generate_variations(shape_idx, base):
    variations = set()
    if shape_idx < 2: variations.add(tuple(sorted(base)))
    else:
        variations.add(tuple(sorted(base)))
        w = max(c for r,c in base)
        variations.add(tuple(sorted([(r, w-c) for r,c in base])))
        if shape_idx >= 3:
             max_r = max(r for r,c in base)
             variations.add(tuple(sorted([(max_r-r, c) for r,c in base])))
             variations.add(tuple(sorted([(max_r-r, w-c) for r,c in base])))
    
    if shape_idx == 4:
        base = [(0,0), (0,1), (0,3), (1,1)]
        variations.add(tuple(sorted(base)))
        variations.add(tuple(sorted([(1-r, c) for r,c in base])))
        
    return [list(v) for v in variations]

def draw_preview(coords):
    if not coords: return ""
    max_r = max(r for r,c in coords)+1; max_c = max(c for r,c in coords)+1
    html = f'<div style="display:grid; grid-template-columns: repeat({max_c}, 15px); gap:2px; justify-content:center; background:#222; padding:10px; border-radius:8px;">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#007acc" if (r,c) in coords else "#333"
            html += f'<div style="width:15px; height:15px; border-radius:2px; background:{bg}"></div>'
    html += '</div>'
    return html

# --- Main App ---

st.title("Chance Analyzer")

with st.sidebar:
    st.header("Upload")
    csv_file = st.file_uploader("Upload CSV", type=None)

df = None
base_shapes = parse_shapes(FIXED_COMBOS_TXT)

if csv_file:
    df, msg = load_data_robust(csv_file)
    if df is None: st.error(msg)

if df is not None:
    req = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    if not all(c in df.columns for c in req):
        if df.shape[1] >= 4:
            df = df.iloc[:, :4]; df.columns = req
        else:
            st.error("Invalid columns"); st.stop()

    grid = df.values
    
    # 1. Settings (Vertical)
    st.write("**Pattern:**")
    s_idx = st.selectbox("Select Pattern", range(len(base_shapes)), format_func=lambda i: PATTERN_NAMES.get(i, f"Pat {i+1}"), label_visibility="collapsed")
    st.markdown(draw_preview(base_shapes[s_idx]), unsafe_allow_html=True)
    
    st.write("---")
    
    st.write("**Select 3 Cards:**")
    unq = np.unique(grid.astype(str))
    opts = sorted([c for c in unq if str(c).lower() != 'nan' and str(c).strip() != ''])
    
    c1 = st.selectbox("1", [""] + opts)
    c2 = st.selectbox("2", [""] + opts)
    c3 = st.selectbox("3", [""] + opts)
    selected = [c for c in [c1,c2,c3] if c!=""]
    
    st.write("")
    if st.button("SEARCH"):
        st.session_state['search_done'] = True
        st.session_state['sel_id'] = None
    
    if st.button("RESET"):
        st.session_state['search_done'] = False
        st.rerun()

    # --- Logic ---
    matches = []
    if st.session_state.get('search_done', False) and len(selected) == 3:
        variations = generate_variations(s_idx, base_shapes[s_idx])
        colors = ['#00ff99', '#ffcc00', '#ff66cc', '#00ccff', '#ff5050']
        
        for shape in variations:
            sh_h = max(r for r,c in shape)+1; sh_w = max(c for r,c in shape)+1
            for r in range(min(len(grid), 51) - sh_h + 1):
                for c in range(4 - sh_w + 1):
                    vals = []; coords = []
                    try:
                        for dr, dc in shape:
                            vals.append(grid[r+dr, c+dc]); coords.append((r+dr, c+dc))
                    except: continue
                    
                    found = 0; used = set()
                    for t in selected:
                        for i, v in enumerate(vals):
                            if i not in used and str(v) == t:
                                used.add(i); found += 1; break
                    
                    if found == 3:
                        miss_i = [i for i in range(4) if i not in used][0]
                        m_id = len(matches) + 1
                        matches.append({
                            'id': m_id, 'miss': vals[miss_i], 'row': coords[miss_i][0],
                            'coords': coords, 'miss_coords': coords[miss_i], 'col': colors[(m_id-1)%len(colors)]
                        })
        matches.sort(key=lambda x: x['row'])

    # --- TABS: Results & Sleeping ---
    st.write("")
    tab1, tab2 = st.tabs(["📋 Results", "💤 Sleeping"])
    
    # TAB 1: RESULTS
    with tab1:
        if st.session_state.get('search_done', False):
            if matches:
                res_df = pd.DataFrame([{'Missing': m['miss'], 'Row': m['row']} for m in matches])
                evt = st.dataframe(res_df, hide_index=True, use_container_width=True, selection_mode="single-row", on_select="rerun")
                if len(evt.selection['rows']) > 0:
                    st.session_state['sel_id'] = matches[evt.selection['rows'][0]]['id']
            else:
                st.info("No matches found")

    # TAB 2: SLEEPING (4 Columns Layout)
    with tab2:
        sleep_cols = st.columns(4)
        icons = ['♣', '♦', '♥', '♠']
        names = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
        colors = ['#aaa', '#ff5555', '#ff5555', '#aaa']
        
        for i in range(4):
            with sleep_cols[i]:
                # Header
                st.markdown(f"""
                <div class="sleep-col">
                    <div class="sleep-header" style="color:{colors[i]}">
                        <div style="font-size:20px">{icons[i]}</div>
                        <div style="font-size:10px">{names[i]}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Logic
                col_d = grid[:, i]
                unq = np.unique(col_d.astype(str))
                lst = []
                for val in unq:
                    if val.lower() == 'nan': continue
                    locs = np.where(col_d == val)[0]
                    if len(locs) > 0 and locs[0] > 7: lst.append((val, locs[0]))
                lst.sort(key=lambda x: x[1], reverse=True)
                
                # Items
                if lst:
                    for val, gap in lst:
                        st.markdown(f'<div class="sleep-item"><b>{val}</b>: {gap}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sleep-item">-</div>', unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

    # --- 3. GAME BOARD ---
    st.write("---")
    st.subheader("📊 Game Board")
    
    sel_match = None
    if st.session_state.get('sel_id'):
        sel_match = next((m for m in matches if m['id'] == st.session_state['sel_id']), None)
        
    style_map = {}
    if sel_match:
        for crd in sel_match['coords']:
            if crd != sel_match['miss_coords']:
                if crd not in style_map: style_map[crd] = ""
                style_map[crd] += f'<div class="frame-box" style="border-color:{sel_match["col"]}; border-width:2px; top:2px; left:2px; right:2px; bottom:2px;"></div>'
        style_map[sel_match['miss_coords']] = "MISS"

    grid_html = '<div class="grid-container">'
    for i in range(4):
        grid_html += f'<div class="grid-header"><div class="suit-icon" style="color:{colors[i]}">{icons[i]}</div>{names[i]}</div>'
        
    for r in range(min(len(grid), 51)):
        for c in range(4):
            val = str(grid[r, c]); 
            if val == 'nan': val = ''
            
            extra = style_map.get((r, c), "")
            inner = val
            if "MISS" in extra:
                inner = f'<div class="missing-circle">{val}</div>'
                extra = ""
            
            grid_html += f'<div class="grid-cell" style="position:relative;">{inner}{extra}</div>'
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)

else:
    st.info("👆 Please upload a file in the sidebar")
