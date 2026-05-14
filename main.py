import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# ==========================================
# 1. PAGE CONFIG & CSS/JS INJECTION
# ==========================================
st.set_page_config(page_title="Chance Analyzer PRO", layout="wide", initial_sidebar_state="expanded")

components.html(
    """
    <script>
    const disableKeyboard = () => {
        const inputs = window.parent.document.querySelectorAll('div[data-baseweb="select"] input');
        inputs.forEach(el => { el.setAttribute('inputmode', 'none'); el.setAttribute('readonly', 'readonly'); });
    };
    disableKeyboard();
    const observer = new MutationObserver(disableKeyboard);
    observer.observe(window.parent.document.body, { childList: true, subtree: true });
    </script>
    """, height=0, width=0
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    .stApp { direction: ltr; text-align: left; background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif;}
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
    
    /* Metrics & Cards */
    div[data-testid="stMetric"] { background: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    
    /* Buttons */
    div.stButton > button { width: 100%; border-radius: 8px; height: 2.8rem; font-weight: 600; transition: all 0.3s ease; border: 1px solid #374151; background: #1F2937; color: #F9FAFB; }
    div.stButton > button:hover { border-color: #3B82F6; box-shadow: 0 0 10px rgba(59, 130, 246, 0.3); }
    div.stButton > button[kind="primary"] { background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); border: none; }
    div.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%); box-shadow: 0 0 15px rgba(59, 130, 246, 0.5); }

    /* Custom Grid Layout */
    .grid-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; background: #111827; padding: 15px; border-radius: 16px; margin-top: 15px; border: 1px solid #1F2937; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4); }
    .grid-cell { background-color: #1F2937; color: #D1D5DB; padding: 0; text-align: center; border-radius: 8px; height: 42px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 15px; position: relative; border: 1px solid #374151; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); transition: all 0.2s ease; }
    
    /* Special Cell States */
    .match-color-0 { background-color: #3B82F6 !important; color: #FFF !important; border-color: #3B82F6 !important; font-weight: 900 !important; }
    .match-color-1 { background-color: #8B5CF6 !important; color: #FFF !important; border-color: #8B5CF6 !important; font-weight: 900 !important; }
    .match-color-2 { background-color: #10B981 !important; color: #FFF !important; border-color: #10B981 !important; font-weight: 900 !important; }
    .match-color-3 { background-color: #F59E0B !important; color: #FFF !important; border-color: #F59E0B !important; font-weight: 900 !important; }
    .match-color-4 { background-color: #EF4444 !important; color: #FFF !important; border-color: #EF4444 !important; font-weight: 900 !important; }
    
    .missing-selected { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important; color: #000 !important; font-weight: 900 !important; border: 1px solid #FFF !important; box-shadow: 0 0 15px rgba(245, 158, 11, 0.8) !important; transform: scale(1.1); z-index: 100; }
    .missing-marker { background-color: rgba(245, 158, 11, 0.15) !important; border: 1px dashed #F59E0B !important; color: #FCD34D !important; }
    .missing-circle { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: #FFFFFF; font-weight: 800; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(245, 158, 11, 0.7); margin: auto; border: 2px solid #FEF3C7; }
    
    .window-highlight { border: 1px solid #3B82F6 !important; box-shadow: inset 0 0 15px rgba(59, 130, 246, 0.2), 0 0 8px rgba(59, 130, 246, 0.3) !important; background-color: #1E3A8A !important; z-index: 5; color: #FFF !important; font-weight: 800;}
    .window-dim { opacity: 0.25 !important; filter: grayscale(60%); }
    
    .awakened-card { background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important; color: #FFF !important; border: 2px solid #34D399 !important; box-shadow: 0 0 15px rgba(16, 185, 129, 0.6) !important; font-weight: 900 !important; transform: scale(1.05); z-index: 10; }
    .history-hit-3row { background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important; color: #FFF !important; border: 2px solid #93C5FD !important; box-shadow: 0 0 15px rgba(59, 130, 246, 0.5) !important; font-weight: 900 !important; z-index: 10; transform: scale(1.05); }

    .grid-header { text-align: center; padding-bottom: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .suit-icon { font-size: 24px; line-height: 1; margin-bottom: 4px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); }
    
    /* Scrollable areas */
    .scrollable-board { max-height: 600px; overflow-y: auto; padding-right: 15px; margin-top: 10px; border: 1px solid #374151; border-radius: 12px; background: #0E1117; }
    .scrollable-board::-webkit-scrollbar { width: 8px; }
    .scrollable-board::-webkit-scrollbar-track { background: #1F2937; border-radius: 8px; }
    .scrollable-board::-webkit-scrollbar-thumb { background: #4B5563; border-radius: 8px; }
    .scrollable-board::-webkit-scrollbar-thumb:hover { background: #6B7280; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONSTANTS & METADATA
# ==========================================
SUITS = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
SUIT_META = {
    'Spades': {'icon': '♠', 'color': '#D1D5DB'},
    'Hearts': {'icon': '♥', 'color': '#EF4444'},
    'Diamonds': {'icon': '♦', 'color': '#EF4444'},
    'Clubs': {'icon': '♣', 'color': '#D1D5DB'}
}

FIXED_COMBOS_TXT = """
A A A A

A
A
A
A

A X X X
X A X X
X X A X
X X X A

A X X X
X A X X
X A X X
X X X A

A A X A
X A X X

A A
A A

A X A
A X A

A X A
X X X
A X A

A X X A
X X X X
X X X X
A X X A

A A A
X A X

A A A
X X X
X A X

A X X X
X A X X
X A A X

A X X X
A X X A
A X X X
"""

PATTERN_NAMES = {
    0: "1. Row", 1: "2. Column", 2: "3. Diagonal", 3: "4. Custom Shape",
    4: "5. Bridge", 5: "6. Square", 6: "7. Parallel Gaps", 7: "8. X-Corners",
    8: "9. Large Corners", 9: "10. T-Shape", 10: "11. T-Spaced", 11: "12. Hook", 12: "13. C-Shape"
}

# ==========================================
# 3. CORE FUNCTIONS (Logic & Data)
# ==========================================
@st.cache_data
def load_data_bulletproof(uploaded_file):
    if uploaded_file is None: return None, "No file"
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp1255')
        except:
            return None, "Error loading file"
            
    df.columns = df.columns.str.strip()
    hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
    df.rename(columns=hebrew_map, inplace=True)
    
    def strict_card_mapper(val):
        v = str(val).upper()
        if '10' in v: return '10'
        for c in ['A', 'K', 'Q', 'J', '9', '8', '7']:
            if c in v: return c
        return ''

    for col in SUITS:
        if col in df.columns:
            df[col] = df[col].apply(strict_card_mapper)
    return df, "ok"

def parse_and_generate_variations(text):
    shapes = []
    for block in text.replace('\r\n', '\n').split('\n\n'):
        if not block.strip(): continue
        coords = [(r, c) for r, line in enumerate(block.strip().split('\n')) for c, char in enumerate(line.split()) if char == 'A']
        if not coords: continue
        min_r, min_c = min(r for r, c in coords), min(c for r, c in coords)
        shapes.append([(r - min_r, c - min_c) for r, c in coords])
    return shapes

BASE_SHAPES = parse_and_generate_variations(FIXED_COMBOS_TXT)

def generate_variations_strict(shape_idx, base_shape):
    def normalize(shape):
        if not shape: return tuple()
        min_r, min_c = min(r for r, c in shape), min(c for r, c in shape)
        return tuple(sorted((r - min_r, c - min_c) for r, c in shape))

    bases = [normalize(base_shape)]
    if shape_idx == 3: bases.append(normalize([(0,0), (0,1), (1,1), (3,3)]))
    if shape_idx == 11: bases.append(normalize([(0,0), (0,1), (1,1), (2,2)]))
        
    variations = set()
    for b in bases:
        variations.add(b)
        if not b: continue
        w, h = max(c for r, c in b), max(r for r, c in b)
        
        transforms = [
            b, [(r, w - c) for r, c in b], [(h - r, c) for r, c in b], [(h - r, w - c) for r, c in b],
            [(c, h - r) for r, c in b], [(w - c, r) for r, c in b], [(c, r) for r, c in b], [(w - c, h - r) for r, c in b]
        ]
        
        valid_transforms = []
        if shape_idx in [0, 1]: valid_transforms = transforms[:1]
        elif shape_idx == 6: valid_transforms = transforms[:4]
        elif shape_idx in [9, 10, 11]: valid_transforms = [transforms[0], transforms[2]]
        elif shape_idx == 12: valid_transforms = [transforms[0], transforms[1]]
        elif shape_idx == 3: valid_transforms = transforms[:4]
        else: valid_transforms = transforms

        for t in valid_transforms:
            norm_t = normalize(t)
            if max(c for r, c in norm_t) < 4:
                variations.add(norm_t)
                
    return [list(v) for v in variations]

def find_matches_for_pattern(shape_idx, selected_cards, grid_data, row_limit):
    found = []
    variations = generate_variations_strict(shape_idx, BASE_SHAPES[shape_idx])
    rows = min(len(grid_data), row_limit)
    
    raw_matches = []
    for shape in variations:
        sh_h = max(r for r, c in shape) + 1
        sh_w = max(c for r, c in shape) + 1
        
        for r in range(rows - sh_h + 1):
            for c in range(4 - sh_w + 1):
                vals, coords = [], []
                try:
                    for dr, dc in shape:
                        vals.append(str(grid_data[r+dr, c+dc]))
                        coords.append((r+dr, c+dc))
                except IndexError: continue
                
                temp_selected = selected_cards.copy()
                used_indices = []
                for i, v in enumerate(vals):
                    if v in temp_selected:
                        temp_selected.remove(v)
                        used_indices.append(i)
                
                if len(used_indices) == 3:
                    miss_i = [i for i in range(len(vals)) if i not in used_indices][0]
                    if shape_idx == 1 and shape[miss_i][0] == 3: continue
                    
                    m_data = {
                        'coords': tuple(sorted(coords)), 
                        'miss_coords': coords[miss_i], 
                        'miss_val': vals[miss_i], 
                        'full_coords_list': coords
                    }
                    if not any(x['coords'] == m_data['coords'] for x in raw_matches):
                        raw_matches.append(m_data)
    
    raw_matches.sort(key=lambda x: x['miss_coords'][0])
    for i, m in enumerate(raw_matches):
        m['id'] = i + 1
        m['color_idx'] = i % 5 
        found.append(m)
    return found

def get_unique_valid(vals):
    seen = set()
    return [v for v in vals if v != "-" and not (v in seen or seen.add(v))]

# ==========================================
# 4. HTML RENDERERS
# ==========================================
def draw_preview_html(shape_coords):
    if not shape_coords: return ""
    max_r, max_c = max(r for r, c in shape_coords) + 1, max(c for r, c in shape_coords) + 1
    html = f'<div style="display:grid; grid-template-columns: repeat({max_c}, 14px); gap: 4px;">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#3B82F6" if (r, c) in shape_coords else "#1F2937"
            border = "1px solid #60A5FA" if (r, c) in shape_coords else "1px solid #374151"
            html += f'<div style="width:14px; height:14px; border-radius:3px; background-color:{bg}; border:{border};"></div>'
    return f'<div style="background: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 12px; display: flex; justify-content: center; align-items: center; height: 100%; box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);">{html}</div>'

def create_sleeping_html_table(data_dict, required_cols):
    max_rows = max([len(v) for v in data_dict.values()] + [0])
    parts = ['<div style="overflow-x: auto; border: 1px solid #1F2937; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"><table style="width: 100%; border-collapse: collapse; font-family: \'Inter\', sans-serif; font-size: 14px;"><thead><tr style="background-color: #111827; border-bottom: 2px solid #374151;">']
    
    for col in required_cols:
        meta = SUIT_META[col]
        parts.append(f'<th style="text-align: center; color: {meta["color"]}; border-right: 1px solid #1F2937; width: 25%;"><div style="padding: 8px 0;"><div style="font-size: 26px; line-height: 1; margin-bottom: 4px;">{meta["icon"]}</div><div style="font-size: 11px; text-transform: uppercase; font-weight: 800; letter-spacing: 1px; color: #9CA3AF;">{col}</div></div></th>')
    parts.append('</tr></thead><tbody>')
    
    for i in range(max_rows):
        parts.append(f'<tr style="background-color: {"#1F2937" if i % 2 == 0 else "#111827"};">')
        for col in required_cols:
            val = data_dict.get(col, [])[i] if i < len(data_dict.get(col, [])) else ""
            color = SUIT_META[col]['color'] if val else "transparent"
            parts.append(f'<td style="padding: 10px; text-align: center; border-right: 1px solid #374151; color: {color}; font-weight: {"600" if val else "normal"}; border-bottom: 1px solid #374151;">{val}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)

def generate_board_html(grid_data, start_row, end_row, cell_classes):
    html = '<div class="grid-container">'
    for col in SUITS:
        meta = SUIT_META[col]
        html += f'<div class="grid-header"><div class="suit-icon" style="color:{meta["color"]};">{meta["icon"]}</div><div style="font-size: 11px; color: #9CA3AF; font-weight: 800; letter-spacing: 1px; text-transform: uppercase;">{col}</div></div>'
    
    for r in range(start_row, min(len(grid_data), end_row)):
        for c in range(4):
            val = str(grid_data[r, c])
            if val == 'nan' or not val: val = ''
            
            classes = "grid-cell " + cell_classes.get((r, c), "")
            inner = val
            
            if "missing-marker" in classes or "missing-selected" in classes:
                inner = f'<div class="missing-circle">{val}</div>'
                classes = classes.replace("missing-marker", "").replace("missing-selected", "")
            
            html += f'<div class="{classes.strip()}">{inner}</div>'
    html += '</div>'
    return html

# ==========================================
# 5. STATE MANAGEMENT
# ==========================================
if 'uploaded_df' not in st.session_state: st.session_state['uploaded_df'] = None
if 'current_shape_idx' not in st.session_state: st.session_state['current_shape_idx'] = 0
if 'window_start' not in st.session_state: st.session_state['window_start'] = 0
if 'chk_all' not in st.session_state: st.session_state['chk_all'] = False
if 'num_combos_val' not in st.session_state: st.session_state['num_combos_val'] = 6

for k in PATTERN_NAMES.keys():
    if f'chk_pat_{k}' not in st.session_state: st.session_state[f'chk_pat_{k}'] = False

def toggle_all():
    val = st.session_state.chk_all
    for k in PATTERN_NAMES.keys(): st.session_state[f'chk_pat_{k}'] = val

# ==========================================
# 6. SIDEBAR & GLOBAL CONTROLS
# ==========================================
st.title("⚡ Chance Analyzer PRO")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Mifal_Hapayis_logo.svg/512px-Mifal_Hapayis_logo.svg.png", width=100)
    st.header("📂 Global Settings")
    csv_file = st.file_uploader("Upload CSV Data", type=None)
    st.markdown("---")
    st.header("⚙️ Algorithm Params")
    global_search_depth = st.number_input("🔍 History Scan Depth", min_value=5, max_value=50000, value=26, step=1)
    global_sleep_thresh = st.number_input("💤 Sleep Threshold", min_value=3, max_value=20, value=7, step=1, help="Number of draws before a card is considered 'Sleeping'")

if csv_file:
    temp_df, msg = load_data_bulletproof(csv_file)
    if temp_df is not None: st.session_state['uploaded_df'] = temp_df

df = st.session_state['uploaded_df']

if df is None:
    st.info("👋 Upload a Chance game CSV file in the sidebar to initialize the engine.")
    st.stop()
    
missing_cols = [c for c in SUITS if c not in df.columns]
if missing_cols:
    st.error(f"Missing columns in data: {missing_cols}")
    st.stop()

grid_data = df[SUITS].values
max_val = max(0, len(grid_data) - 3)

def dec_win(): st.session_state['window_start'] = max(0, st.session_state['window_start'] - 1)
def inc_win(): st.session_state['window_start'] = min(max_val, st.session_state['window_start'] + 1)
def prev_pat(): st.session_state['current_shape_idx'] = (st.session_state['current_shape_idx'] - 1) % len(PATTERN_NAMES)
def next_pat(): st.session_state['current_shape_idx'] = (st.session_state['current_shape_idx'] + 1) % len(PATTERN_NAMES)
def reset_search(): st.session_state['search_done'] = False

# ==========================================
# 7. MAIN DASHBOARD TABS
# ==========================================
tab_pred, tab_matches, tab_lab, tab_hist_pred, tab_sleepers = st.tabs([
    "🔮 SMART PREDICTOR", 
    "📋 PATTERN MATCHES", 
    "🧪 PATTERN LAB",
    "📈 3-ROW BACKTEST",
    "💤 SLEEPERS (LIVE & HISTORY)"
])

# ------------------------------------------
# TAB 1: SMART PREDICTOR
# ------------------------------------------
with tab_pred:
    st.markdown("<h3 style='color: #F9FAFB; margin-bottom: 5px;'>Smart 3-Row Predictor</h3>", unsafe_allow_html=True)
    
    c_nav1, c_nav2, c_nav3 = st.columns([1, 2, 1])
    with c_nav1: st.button("➖ Older Draw", use_container_width=True, on_click=inc_win, key="btn_inc_pred") 
    with c_nav2: st.markdown(f"<div style='text-align:center; font-size: 18px; font-weight: 800; background: #1F2937; padding: 5px; border-radius: 8px; border: 1px solid #3B82F6; color: #60A5FA; box-shadow: 0 0 10px rgba(59,130,246,0.2);'>Target Row: {st.session_state['window_start']}</div>", unsafe_allow_html=True)
    with c_nav3: st.button("➕ Newer Draw", use_container_width=True, on_click=dec_win, key="btn_dec_pred")
            
    win_start = st.session_state['window_start']

    with st.expander("⚙️ Pattern Selection Engine", expanded=True):
        st.checkbox("✅ Select / Deselect All", key="chk_all", on_change=toggle_all)
        st.markdown("<hr style='margin: 8px 0; border-color: #374151;'>", unsafe_allow_html=True)
        active_pats = []
        cols = st.columns(3)
        for i, (k, v) in enumerate(PATTERN_NAMES.items()):
            if cols[i % 3].checkbox(v, key=f"chk_pat_{k}"): active_pats.append(k)
                
    w_data = grid_data[win_start : win_start + 3, :]
    has_actual = win_start > 0
    actual_row = grid_data[win_start - 1, :] if has_actual else [None]*4
    historical_scan = grid_data[win_start : win_start + global_search_depth]
    
    if has_actual:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E3A8A 0%, #111827 100%); border: 1px solid #3B82F6; border-radius: 12px; padding: 15px; margin: 15px 0; text-align: center; box-shadow: 0 4px 10px rgba(59,130,246,0.2);">
            <div style="font-size: 13px; color: #93C5FD; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">🎯 Actual Outcome (Row {win_start - 1})</div>
            <div style="font-size: 24px; font-weight: 900; color: #FFFFFF; letter-spacing: 3px;">
                <span style="color: #D1D5DB;">♠ {actual_row[0]}</span> <span style="color:#4B5563;">|</span> 
                <span style="color: #EF4444;">♥ {actual_row[1]}</span> <span style="color:#4B5563;">|</span> 
                <span style="color: #EF4444;">♦ {actual_row[2]}</span> <span style="color:#4B5563;">|</span> 
                <span style="color: #D1D5DB;">♣ {actual_row[3]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #064E3B 0%, #111827 100%); border: 1px solid #10B981; border-radius: 12px; padding: 15px; margin: 15px 0; text-align: center;">
            <div style="font-size: 13px; color: #6EE7B7; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">🔮 Predicting Next Draw</div>
            <div style="font-size: 16px; font-weight: 600; color: #D1FAE5;">Awaiting actual results...</div>
        </div>
        """, unsafe_allow_html=True)

    suit_tabs = st.tabs([f"{SUIT_META[s]['icon']} {s}" for s in SUITS])
    suit_preds = {}
    
    for i, suit in enumerate(SUITS):
        with suit_tabs[i]:
            triplet = [str(w_data[r, i]) for r in range(min(3, len(w_data))) if str(w_data[r, i]).strip() and str(w_data[r, i]).lower() != 'nan']
            actual_card = str(actual_row[i]).strip().upper() if has_actual else "-"
            
            html_table = f"<div style='text-align: center; color: #8B949E; font-size: 13px; font-weight: 600; margin-bottom: 10px;'>Base Triplet: <span style='color: #D1D5DB;'>[{' | '.join(triplet) if len(triplet)==3 else 'Incomplete'}]</span></div>"
            html_table += "<table style='max-width: 400px; margin: 0 auto; width: 100%; border-collapse: collapse; text-align: center; background: #161B22; border-radius: 10px; overflow: hidden; border: 1px solid #30363D;'><thead style='background: #21262D; color: #8B949E; font-size: 11px; text-transform: uppercase;'><tr><th style='padding: 8px; border-bottom: 1px solid #30363D; width: 60%;'>Predicted Card</th><th style='padding: 8px; border-bottom: 1px solid #30363D; width: 40%;'>Matches</th></tr></thead><tbody>"
            
            suit_preds[suit] = []
            if len(triplet) == 3 and active_pats:
                all_missing = []
                for p_idx in active_pats:
                    m = find_matches_for_pattern(p_idx, triplet, historical_scan, global_search_depth)
                    all_missing.extend([x['miss_val'].strip().upper() for x in m])
                
                if all_missing:
                    counts = pd.Series(all_missing).value_counts()
                    suit_preds[suit] = list(counts.index)
                    for card, count in counts.items():
                        is_hit = has_actual and (card == actual_card)
                        row_bg = "background: rgba(16, 185, 129, 0.1);" if is_hit else ""
                        border_left = "border-left: 3px solid #10B981;" if is_hit else "border-left: 3px solid transparent;"
                        card_color = "#10B981" if is_hit else "#58A6FF"
                        icon = "<span style='background: #10B981; color: white; padding: 2px 6px; border-radius: 4px; font-size: 9px; margin-left: 8px;'>HIT</span>" if is_hit else ""
                        html_table += f"<tr style='{row_bg}'><td style='padding: 8px; {border_left} font-weight: 900; font-size: 18px; color: {card_color}; border-bottom: 1px solid #30363D;'>{card}{icon}</td><td style='padding: 8px; font-weight: 800; color: #FAFAFA; border-bottom: 1px solid #30363D;'>{count}</td></tr>"
                else: html_table += "<tr><td style='padding: 8px; color: #F85149; font-weight: 600;'>No Match</td><td style='padding: 8px; color: #8B949E;'>0</td></tr>"
            else:
                msg = "Select patterns above" if len(triplet) == 3 else "Incomplete Triplet"
                html_table += f"<tr><td colspan='2' style='padding: 15px; color: #FCD34D; font-weight: 600;'>{msg}</td></tr>"
            html_table += "</tbody></table>"
            st.markdown(html_table, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("<h3 style='margin: 0; color: #FAFAFA;'>🎲 Auto-Generated Tickets (Safety Net)</h3>", unsafe_allow_html=True)
    num_combos = st.slider("Select Number of Tickets", 1, 10, st.session_state['num_combos_val'], label_visibility="collapsed")
    st.session_state['num_combos_val'] = num_combos
    
    def get_c(s, r): return suit_preds.get(s, [])[r] if len(suit_preds.get(s, [])) > r else ("-" if len(suit_preds.get(s, [])) == 0 else suit_preds.get(s, [])[0])
        
    combos = [
        {"name": "Ticket 1 (Drop ♠) [Safe ♦]", "cfg": [ ["-"], [get_c('Hearts',0)], [get_c('Diamonds',1), get_c('Diamonds',0)], [get_c('Clubs',0)] ]},
        {"name": "Ticket 2 (Drop ♥) [Safe ♠]", "cfg": [ [get_c('Spades',1), get_c('Spades',0)], ["-"], [get_c('Diamonds',0)], [get_c('Clubs',0)] ]},
        {"name": "Ticket 3 (Drop ♦) [Safe ♥]", "cfg": [ [get_c('Spades',0)], [get_c('Hearts',1), get_c('Hearts',0)], ["-"], [get_c('Clubs',0)] ]},
        {"name": "Ticket 4 (Drop ♣) [Safe ♦]", "cfg": [ [get_c('Spades',0)], [get_c('Hearts',0)], [get_c('Diamonds',1), get_c('Diamonds',0)], ["-"] ]},
        {"name": "Ticket 5 (Drop ♠) [Safe ♣]", "cfg": [ ["-"], [get_c('Hearts',0)], [get_c('Diamonds',0)], [get_c('Clubs',1), get_c('Clubs',0)] ]},
        {"name": "Ticket 6 (Drop ♥) [Safe ♦]", "cfg": [ [get_c('Spades',0)], ["-"], [get_c('Diamonds',2), get_c('Diamonds',0)], [get_c('Clubs',0)] ]},
        {"name": "Ticket 7 (Drop ♦) [Safe ♣]", "cfg": [ [get_c('Spades',0)], [get_c('Hearts',0)], ["-"], [get_c('Clubs',2), get_c('Clubs',0)] ]},
        {"name": "Ticket 8 (Drop ♣) [Safe ♠]", "cfg": [ [get_c('Spades',2), get_c('Spades',0)], [get_c('Hearts',0)], [get_c('Diamonds',0)], ["-"] ]},
        {"name": "Ticket 9 (Drop ♠) [Mix A]",  "cfg": [ ["-"], [get_c('Hearts',2), get_c('Hearts',0)], [get_c('Diamonds',1)], [get_c('Clubs',0)] ]},
        {"name": "Ticket 10 (Drop ♥) [Mix B]", "cfg": [ [get_c('Spades',1)], ["-"], [get_c('Diamonds',2), get_c('Diamonds',0)], [get_c('Clubs',0)] ]}
    ]
    
    sel_combos = combos[:num_combos]
    total_cost = sum([np.prod([len(get_unique_valid(v)) for v in cb["cfg"] if get_unique_valid(v)]) * 5 for cb in sel_combos]) if active_pats else 0

    if not active_pats: st.info("👆 Please select patterns to generate tickets.")
    else:
        st.markdown(f"<div style='background: #111827; border: 1px solid #10B981; border-radius: 8px; padding: 10px; margin-bottom: 15px; color: #10B981; font-weight: 800; font-size: 18px; text-align: center;'>Total Investment: ₪{total_cost}</div>", unsafe_allow_html=True)
        html_c = '<div style="display: flex; flex-wrap: wrap; gap: 15px;">'
        for cb in sel_combos:
            html_c += f'<div style="flex: 1 1 300px; background: #161B22; border: 1px solid #30363D; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);"><div style="color: #FCD34D; font-weight: 800; font-size: 14px; border-bottom: 1px solid #30363D; padding-bottom: 8px; margin-bottom: 10px; text-align: center;">{cb["name"]}</div><div style="display: flex; justify-content: space-between;">'
            for i, s in enumerate(SUITS):
                u_vals = get_unique_valid(cb["cfg"][i])
                val_str = " + ".join(u_vals) if u_vals else "-"
                color = SUIT_META[s]['color'] if val_str != "-" else "#4B5563"
                bg = "background: rgba(59, 130, 246, 0.1); border: 1px dashed #3B82F6;" if len(u_vals)>1 else ("background: #111827; border: 1px dashed #374151; opacity: 0.5;" if val_str=="-" else "background: #111827; border: 1px solid #374151;")
                html_c += f'<div style="text-align: center; padding: 8px; border-radius: 8px; {bg} width: 23%;"><div style="color: {color}; font-size: 18px; margin-bottom: 4px;">{SUIT_META[s]["icon"]}</div><div style="color: #FFF; font-weight: 900; font-size: 14px;">{val_str}</div></div>'
            html_c += '</div></div>'
        st.markdown(html_c + '</div>', unsafe_allow_html=True)
            
    st.markdown("<h4 style='margin-top: 25px; font-weight: 800; color: #FAFAFA;'>Target Context Board</h4>", unsafe_allow_html=True)
    c_class = {(r, c): "window-highlight" if win_start <= r < win_start + 3 else "window-dim" for r in range(max(0, win_start - 1), min(len(grid_data), win_start + 30)) for c in range(4)}
    st.markdown(generate_board_html(grid_data, max(0, win_start - 1), win_start + 30, c_class), unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: PATTERN MATCHES
# ------------------------------------------
with tab_matches:
    with st.expander("⚙️ Configuration & Target Inputs", expanded=not st.session_state.get('search_done', False)):
        col_conf, col_prev = st.columns([4, 1])
        with col_conf:
            nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
            with nav_col1: st.button("◀", use_container_width=True, key="prev_pat_btn", on_click=prev_pat)
            with nav_col2:
                curr_name = PATTERN_NAMES[st.session_state['current_shape_idx']]
                st.markdown(f"<div style='display: flex; align-items: center; justify-content: center; height: 2.8rem; background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; font-weight: 600; color: #58A6FF; font-size: 15px;'>{curr_name}</div>", unsafe_allow_html=True)
            with nav_col3: st.button("▶", use_container_width=True, key="next_pat_btn", on_click=next_pat)
            shape_idx = st.session_state['current_shape_idx']

        with col_prev: st.markdown(draw_preview_html(BASE_SHAPES[shape_idx]), unsafe_allow_html=True)
        
        raw_cards = np.unique(grid_data.astype(str))
        clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
        
        c1, c2, c3 = st.columns(3)
        with c1: card1 = st.selectbox("Card 1", [""] + clean_cards, key="c1", label_visibility="collapsed")
        with c2: card2 = st.selectbox("Card 2", [""] + clean_cards, key="c2", label_visibility="collapsed")
        with c3: card3 = st.selectbox("Card 3", [""] + clean_cards, key="c3", label_visibility="collapsed")
        sel_cards = [c for c in [card1, card2, card3] if c != ""]
        
        b_search, b_empty, b_reset = st.columns([3, 3, 1])
        with b_search: run_search = st.button("🔍 Search", type="primary", use_container_width=True)
        with b_reset: st.button("Reset", use_container_width=True, on_click=reset_search)

    found_matches = []
    if (run_search or st.session_state.get('search_done', False)) and len(sel_cards) == 3:
        st.session_state['search_done'] = True
        found_matches = find_matches_for_pattern(st.session_state.get('current_shape_idx', shape_idx), sel_cards, grid_data, global_search_depth)

    if found_matches:
        raw_df = pd.DataFrame([{'Missing Card': m['miss_val'], 'Row': m['miss_coords'][0], 'Hidden_ID': m['id']} for m in found_matches])
        grouped_df = raw_df.groupby('Missing Card').agg({'Row': lambda x: sorted(list(x)), 'Hidden_ID': list}).reset_index()
        grouped_df['Count'] = grouped_df['Hidden_ID'].apply(len)
        grouped_df = grouped_df.sort_values(by='Count', ascending=False)
        grouped_df['Count'] = grouped_df['Count'].astype(str)
        grouped_df['Row Indexes'] = grouped_df['Row'].apply(lambda x: ", ".join(map(str, x)))
        
        display_df = grouped_df[['Missing Card', 'Count', 'Row Indexes', 'Hidden_ID']]
        event = st.dataframe(display_df.drop(columns=['Hidden_ID']), hide_index=True, use_container_width=True, selection_mode="single-row", on_select="rerun")
        selected_match_ids = display_df.iloc[event.selection['rows'][0]]['Hidden_ID'] if len(event.selection['rows']) > 0 else None
        
        st.markdown("<h4 style='margin-top: 15px; font-weight: 800; color: #F3F4F6;'>Live Game Board</h4>", unsafe_allow_html=True)
        cell_classes = {}
        matches_to_show = [m for m in found_matches if m['id'] in selected_match_ids] if selected_match_ids else found_matches

        for m in matches_to_show:
            for coord in m['full_coords_list']:
                if coord != m['miss_coords']: cell_classes[coord] = cell_classes.get(coord, "") + f" match-color-{m['color_idx']}"
            miss = m['miss_coords']
            cell_classes[miss] = cell_classes.get(miss, "") + (" missing-selected" if selected_match_ids else " missing-marker")
            
        draw_limit = max(30, max(c[0] for m in matches_to_show for c in m['full_coords_list']) + 3) if matches_to_show else 30
        st.markdown(generate_board_html(grid_data, 0, draw_limit, cell_classes), unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: PATTERN LAB (Manual Search)
# ------------------------------------------
with tab_lab:
    st.markdown("<h3 style='color: #F9FAFB;'>🧪 Pattern Lab (Manual Exploration)</h3>", unsafe_allow_html=True)
    col_nav, col_prev = st.columns([3, 1])
    with col_nav:
        st.selectbox("Select Shape", list(PATTERN_NAMES.values()), index=st.session_state['current_shape_idx'], key='lab_shape_sel', disabled=True)
        c1, c2 = st.columns(2)
        c1.button("◀ Previous Shape", on_click=prev_pat, use_container_width=True, key="lab_prev")
        c2.button("Next Shape ▶", on_click=next_pat, use_container_width=True, key="lab_next")
        
        cards = sorted([c for c in np.unique(grid_data.astype(str)) if str(c).strip() and str(c).lower()!='nan'])
        sc1, sc2, sc3 = st.columns(3)
        with sc1: c_1 = st.selectbox("Card 1", [""] + cards, key="l_c1")
        with sc2: c_2 = st.selectbox("Card 2", [""] + cards, key="l_c2")
        with sc3: c_3 = st.selectbox("Card 3", [""] + cards, key="l_c3")
        sel_cards = [c for c in [c_1, c_2, c_3] if c]
        
    with col_prev: st.markdown(draw_preview_html(BASE_SHAPES[st.session_state['current_shape_idx']]), unsafe_allow_html=True)

    if st.button("🔍 Search History", type="primary"):
        if len(sel_cards) == 3:
            matches = find_matches_for_pattern(st.session_state['current_shape_idx'], sel_cards, grid_data, global_search_depth)
            if matches:
                lab_classes = {}
                for m in matches:
                    for coord in m['full_coords_list']:
                        if coord != m['miss_coords']: lab_classes[coord] = lab_classes.get(coord, "") + f" match-color-{m['color_idx']}"
                    lab_classes[m['miss_coords']] = lab_classes.get(m['miss_coords'], "") + " missing-selected"
                st.markdown(f"**Found {len(matches)} matches.**")
                st.markdown(generate_board_html(grid_data, 0, max(c[0] for m in matches for c in m['full_coords_list']) + 3, lab_classes), unsafe_allow_html=True)
            else: st.warning("No matches found.")
        else: st.warning("Please select exactly 3 cards.")

# ------------------------------------------
# TAB 4: 3-ROW BACKTEST
# ------------------------------------------
with tab_hist_pred:
    st.markdown("<h3 style='color: #F9FAFB;'>📈 Backtest: Smart Predictor</h3>", unsafe_allow_html=True)
    st.markdown("Evaluates the last 200 draws. <span style='color:#60A5FA; font-weight:bold;'>BLUE rows</span> represent draws where the **Top 3 predicted cards** matched the actual outcome in **3 or 4 suits**.", unsafe_allow_html=True)
    
    if st.button("🚀 Run Predictor Backtest", key="btn_run_bt_pred"):
        with st.spinner("Crunching historical data..."):
            bt_classes = {(r, c): "window-dim" for r in range(min(200, len(grid_data))) for c in range(4)}
            active_pats = [k for k in PATTERN_NAMES.keys() if st.session_state.get(f"chk_pat_{k}", False)]
            wins = 0
            
            if not active_pats: st.error("Please enable patterns in the SMART PREDICTOR tab first.")
            else:
                for r in range(1, min(201, len(grid_data)-2)):
                    hit_count = 0
                    for c in range(4):
                        triplet = [str(grid_data[x, c]) for x in range(r, r+3)]
                        if "" in triplet or "nan" in triplet or "NAN" in triplet or len(triplet) < 3: continue
                        
                        misses = []
                        for p in active_pats:
                            misses.extend([x['miss_val'].strip().upper() for x in find_matches_for_pattern(p, triplet, grid_data[r:], global_search_depth)])
                        
                        if misses:
                            top_3 = list(pd.Series(misses).value_counts().index)[:3]
                            if str(grid_data[r-1, c]).strip().upper() in top_3: hit_count += 1
                    
                    if hit_count >= 3:
                        wins += 1
                        for c in range(4): bt_classes[(r-1, c)] = "history-hit-3row"
                
                st.session_state['bt_pred_classes'], st.session_state['bt_pred_wins'], st.session_state['bt_pred_run'] = bt_classes, wins, True
    
    if st.session_state.get('bt_pred_run', False):
        col1, col2 = st.columns(2)
        col1.metric("🎯 Total Winning Draws (3+ Hits)", st.session_state['bt_pred_wins'])
        col2.metric("📊 Win Rate (approx)", f"{(st.session_state['bt_pred_wins'] / min(200, len(grid_data)-1)) * 100:.1f}%")
        st.markdown(f'<div class="scrollable-board">{generate_board_html(grid_data, 0, min(200, len(grid_data)), st.session_state["bt_pred_classes"])}</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 5: SLEEPERS (LIVE & HISTORY)
# ------------------------------------------
with tab_sleepers:
    st.markdown("<h3 style='color: #F9FAFB;'>💤 Sleepers Radar & History</h3>", unsafe_allow_html=True)
    st.markdown(f"Using Global Sleep Threshold: **{global_sleep_thresh}** draws.")
    
    # 1. LIVE RADAR
    st.markdown("<h4 style='color: #10B981; margin-top: 15px;'>Live Radar: Sleeping Right Now (Heading into Next Draw)</h4>", unsafe_allow_html=True)
    live_sleep_data = {}
    for col_idx, col_name in enumerate(SUITS):
        col_data = grid_data[:, col_idx]
        lst = []
        for c in np.unique(col_data.astype(str)):
            if not str(c).strip() or str(c).lower()=='nan': continue
            locs = np.where(col_data == str(c))[0]
            gap = locs[0] if len(locs) > 0 else len(col_data)
            if gap >= global_sleep_thresh: lst.append((c, gap))
        lst.sort(key=lambda x: x[1], reverse=True)
        live_sleep_data[col_name] = [f"{item[0]} : {item[1]}" for item in lst]
        
    if any(live_sleep_data.values()): st.markdown(create_sleeping_html_table(live_sleep_data, SUITS), unsafe_allow_html=True)
    else: st.info("No sleeping cards currently.")
        
    st.markdown("---")
    
    # 2. HISTORICAL BACKTEST & TIME MACHINE
    st.markdown("<h3 style='color: #F9FAFB;'>📉 Historical Backtest (Last 200 Draws)</h3>", unsafe_allow_html=True)
    
    if st.button("🚀 Analyze Past 200 Draws", key="btn_run_slp_hist"):
        with st.spinner("Analyzing chain reactions..."):
            bt_slp_classes = {(r, c): "window-dim" for r in range(min(200, len(grid_data))) for c in range(4)}
            wins_sleep = 0
            hist_rows = []
            
            for r in range(min(200, len(grid_data)-1)):
                row_awakened = []
                for c in range(4):
                    val = str(grid_data[r, c])
                    if not val or val.lower() == 'nan': continue
                    col_data = grid_data[r+1:, c]
                    locs = np.where(col_data == val)[0]
                    gap = locs[0] if len(locs) > 0 else len(col_data)
                    if gap >= global_sleep_thresh: row_awakened.append((r, c))
                
                # Append to interactive dataframe
                hist_rows.append({
                    "Row": r,
                    "♠ Spades": str(grid_data[r, 0]).replace('nan','-'),
                    "♥ Hearts": str(grid_data[r, 1]).replace('nan','-'),
                    "♦ Diamonds": str(grid_data[r, 2]).replace('nan','-'),
                    "♣ Clubs": str(grid_data[r, 3]).replace('nan','-'),
                    "Awakenings": len(row_awakened)
                })
                
                if len(row_awakened) >= 3:
                    wins_sleep += 1
                    for cell in row_awakened: bt_slp_classes[cell] = "awakened-card"
                    for c in range(4): 
                        if (r,c) not in bt_slp_classes: bt_slp_classes[(r,c)] = "" # Un-dim the rest of the winning row
            
            st.session_state['bt_slp_classes'], st.session_state['bt_slp_wins'] = bt_slp_classes, wins_sleep
            st.session_state['sleep_hist_df'] = pd.DataFrame(hist_rows)
            st.session_state['sleep_hist_run'] = True

    if st.session_state.get('sleep_hist_run', False):
        st.metric("🌪️ Mass Awakenings (3+ Cards)", st.session_state['bt_slp_wins'])
        st.markdown(f'<div class="scrollable-board">{generate_board_html(grid_data, 0, min(200, len(grid_data)), st.session_state["bt_slp_classes"])}</div>', unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #F9FAFB; margin-top: 30px;'>🔍 Interactive Time Machine</h4>", unsafe_allow_html=True)
        st.markdown("Select any row in the table below to see the exact state of sleeping cards **just before** that draw occurred.")
        
        df_to_show = st.session_state['sleep_hist_df']
        event = st.dataframe(df_to_show, use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")
        
        if len(event.selection['rows']) > 0:
            target_row = df_to_show.iloc[event.selection['rows'][0]]['Row']
            st.markdown(f"<h5 style='color: #10B981; margin-top: 15px;'>Sleepers Just Before Row {target_row}</h5>", unsafe_allow_html=True)
            
            calc_start = target_row + 1
            tm_sleep_data = {}
            for col_idx, col_name in enumerate(SUITS):
                col_data = grid_data[calc_start:, col_idx] if calc_start < len(grid_data) else np.array([])
                lst = []
                for c in np.unique(grid_data[:, col_idx].astype(str)):
                    if not str(c).strip() or str(c).lower()=='nan': continue
                    locs = np.where(col_data == str(c))[0]
                    gap = locs[0] if len(locs) > 0 else len(col_data)
                    if gap >= global_sleep_thresh: lst.append((c, gap))
                lst.sort(key=lambda x: x[1], reverse=True)
                tm_sleep_data[col_name] = [f"{item[0]} : {item[1]}" for item in lst]
                
            if any(tm_sleep_data.values()): st.markdown(create_sleeping_html_table(tm_sleep_data, SUITS), unsafe_allow_html=True)
            else: st.info("No sleeping cards found for this historical draw.")
