import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Chance Analyzer PRO",
    layout="wide",
    initial_sidebar_state="collapsed"
)

components.html(
    """
    <script>
    const doc = window.parent.document;
    const disableKeyboard = () => {
        const inputs = doc.querySelectorAll('div[data-baseweb="select"] input');
        inputs.forEach(el => {
            el.setAttribute('inputmode', 'none');
            el.setAttribute('readonly', 'readonly');
        });
    };
    disableKeyboard();
    const observer = new MutationObserver(disableKeyboard);
    observer.observe(doc.body, { childList: true, subtree: true });
    </script>
    """,
    height=0, width=0
)

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
    0: "1. Row",
    1: "2. Column",
    2: "3. Diagonal",
    3: "4. Custom Shape",
    4: "5. Bridge",
    5: "6. Square",
    6: "7. Parallel Gaps",
    7: "8. X-Corners",
    8: "9. Large Corners",
    9: "10. T-Shape",
    10: "11. T-Spaced",
    11: "12. Hook",
    12: "13. C-Shape"
}

def parse_shapes_strict(text):
    shapes = []
    text = text.replace('\r\n', '\n')
    blocks = text.split('\n\n')
    for block in blocks:
        if not block.strip(): continue
        lines = block.strip().split('\n')
        coords = []
        for r, line in enumerate(lines):
            tokens = line.split()
            for c, char in enumerate(tokens):
                if char == 'A':
                    coords.append((r, c))
        if not coords: continue
        
        min_r = min(r for r, c in coords)
        min_c = min(c for r, c in coords)
        normalized = [(r - min_r, c - min_c) for r, c in coords]
        shapes.append(normalized)
    return shapes

def normalize_shape(shape):
    if not shape: return tuple()
    min_r = min(r for r, c in shape)
    min_c = min(c for r, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))

def generate_variations_strict(shape_idx, base_shape):
    bases = [normalize_shape(base_shape)]
    
    if shape_idx == 3:
        bases.append(normalize_shape([(0,0), (0,1), (1,1), (3,3)]))
    if shape_idx == 11:
        bases.append(normalize_shape([(0,0), (0,1), (1,1), (2,2)]))
        
    variations = set()
    for b in bases:
        variations.add(b)
        if not b: continue
        
        w = max(c for r, c in b)
        h = max(r for r, c in b)
        
        mirror_h = normalize_shape([(r, w - c) for r, c in b])
        flip_v = normalize_shape([(h - r, c) for r, c in b])
        rot_180 = normalize_shape([(h - r, w - c) for r, c in b])
        
        rot_90 = normalize_shape([(c, h - r) for r, c in b])
        rot_270 = normalize_shape([(w - c, r) for r, c in b])
        transp1 = normalize_shape([(c, r) for r, c in b]) 
        transp2 = normalize_shape([(w - c, h - r) for r, c in b]) 
        
        if shape_idx in [0, 1]:
            pass 
        elif shape_idx == 6:
            variations.update([mirror_h, flip_v, rot_180])
        elif shape_idx in [9, 10, 11]:
            variations.update([flip_v])
        elif shape_idx == 12:
            variations.update([mirror_h])
        elif shape_idx == 3:
            variations.update([mirror_h, flip_v, rot_180])
        else:
            variations.update([mirror_h, flip_v, rot_180, rot_90, rot_270, transp1, transp2])
            
    valid_variations = []
    for v in variations:
        if max(c for r, c in v) < 4:
            valid_variations.append(list(v))
            
    return valid_variations

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .stApp { direction: ltr; text-align: left; background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif;}
    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
    .stSelectbox, .stMultiSelect, div[data-testid="stExpander"] { direction: ltr; text-align: left; }
    div[data-baseweb="select"] > div { background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; }
    
    div[data-baseweb="select"] input {
        pointer-events: none !important;
        user-select: none !important;
        -webkit-user-select: none !important;
        touch-action: none !important;
    }

    div.stButton > button { width: 100%; border-radius: 8px; height: 2.8rem; font-weight: 600; transition: all 0.3s ease; border: 1px solid #374151; background: #1F2937; color: #F9FAFB; }
    div.stButton > button:hover { border-color: #3B82F6; box-shadow: 0 0 10px rgba(59, 130, 246, 0.3); }
    div.stButton > button[kind="primary"] { background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); border: none; }
    div.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%); box-shadow: 0 0 15px rgba(59, 130, 246, 0.5); }

    .grid-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; background: #111827; padding: 12px; border-radius: 16px; margin-top: 15px; border: 1px solid #1F2937; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); }
    .grid-cell { background-color: #1F2937; color: #D1D5DB; padding: 0; text-align: center; border-radius: 8px; height: 42px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 15px; position: relative; border: 1px solid #374151; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); transition: all 0.2s ease; }
    
    .match-color-0 { background-color: #3B82F6 !important; color: #FFF !important; border-color: #3B82F6 !important; font-weight: 900 !important; }
    .match-color-1 { background-color: #8B5CF6 !important; color: #FFF !important; border-color: #8B5CF6 !important; font-weight: 900 !important; }
    .match-color-2 { background-color: #10B981 !important; color: #FFF !important; border-color: #10B981 !important; font-weight: 900 !important; }
    .match-color-3 { background-color: #F59E0B !important; color: #FFF !important; border-color: #F59E0B !important; font-weight: 900 !important; }
    .match-color-4 { background-color: #EF4444 !important; color: #FFF !important; border-color: #EF4444 !important; font-weight: 900 !important; }
    
    .missing-selected { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important; color: #000 !important; font-weight: 900 !important; border: 1px solid #FFF !important; box-shadow: 0 0 15px rgba(245, 158, 11, 0.8) !important; transform: scale(1.1); z-index: 100; }
    .missing-marker { background-color: rgba(245, 158, 11, 0.15) !important; border: 1px dashed #F59E0B !important; color: #FCD34D !important; }
    .missing-circle { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: #FFFFFF; font-weight: 800; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(245, 158, 11, 0.7); margin: auto; border: 2px solid #FEF3C7; }
    
    .window-highlight { border: 1px solid #F59E0B !important; box-shadow: inset 0 0 15px rgba(245, 158, 11, 0.5), 0 0 8px rgba(245, 158, 11, 0.3) !important; background-color: #1F2937 !important; z-index: 5; }
    .window-dim { opacity: 0.3 !important; filter: grayscale(40%); }
    
    .awakened-card { background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important; color: #FFF !important; border: 2px solid #34D399 !important; box-shadow: 0 0 15px rgba(16, 185, 129, 0.6) !important; font-weight: 900 !important; transform: scale(1.05); z-index: 10; }
    .trigger-row { background-color: rgba(59, 130, 246, 0.1) !important; border-left: 2px solid #3B82F6 !important; border-right: 2px solid #3B82F6 !important; }

    .grid-header { text-align: center; padding-bottom: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .suit-icon { font-size: 24px; line-height: 1; margin-bottom: 4px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); }
    .shape-preview-wrapper { background: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 12px; display: flex; justify-content: center; align-items: center; height: 100%; box-shadow: inset 0 2px 10px rgba(0,0,0,0.2); }
    
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #1F2937; }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { text-align: left !important; }
    
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem; }
    div[data-testid="stHorizontalBlock"] { align-items: center; }
</style>
""", unsafe_allow_html=True)

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
        if 'A' in v: return 'A'
        if 'K' in v: return 'K'
        if 'Q' in v: return 'Q'
        if 'J' in v: return 'J'
        if '9' in v: return '9'
        if '8' in v: return '8'
        if '7' in v: return '7'
        return ''

    required_cols = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    for col in required_cols:
        if col in df.columns:
            df[col] = df[col].apply(strict_card_mapper)
            
    return df, "ok"

def draw_preview_html(shape_coords):
    if not shape_coords: return ""
    max_r = max(r for r, c in shape_coords) + 1
    max_c = max(c for r, c in shape_coords) + 1
    
    grid_html = f'<div style="display:grid; grid-template-columns: repeat({max_c}, 14px); gap: 4px;">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#3B82F6" if (r, c) in shape_coords else "#1F2937"
            border = "1px solid #60A5FA" if (r, c) in shape_coords else "1px solid #374151"
            box_shadow = "box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);" if (r, c) in shape_coords else ""
            grid_html += f'<div style="width:14px; height:14px; border-radius:3px; background-color:{bg}; border:{border}; {box_shadow}"></div>'
    grid_html += '</div>'
    return f'<div class="shape-preview-wrapper">{grid_html}</div>'

def create_sleeping_html_table(data_dict, required_cols):
    meta = {
        'Clubs': {'icon': '♣', 'color': '#D1D5DB'},    
        'Diamonds': {'icon': '♦', 'color': '#EF4444'}, 
        'Hearts': {'icon': '♥', 'color': '#EF4444'},   
        'Spades': {'icon': '♠', 'color': '#D1D5DB'}
    }
    
    max_rows = 0
    clean_data = {}
    for col in required_cols:
        clean_data[col] = data_dict.get(col, [])
        if len(clean_data[col]) > max_rows: max_rows = len(clean_data[col])
            
    parts = ['<div style="overflow-x: auto; border: 1px solid #1F2937; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">',
             '<table style="width: 100%; border-collapse: collapse; font-family: \'Inter\', sans-serif; font-size: 14px;">',
             '<thead><tr style="background-color: #111827; border-bottom: 2px solid #374151;">']
    
    for col in required_cols:
        c_meta = meta.get(col, {'icon': '', 'color': '#fff'})
        header_content = f'<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 8px 0;"><div style="font-size: 26px; line-height: 1; margin-bottom: 4px;">{c_meta["icon"]}</div><div style="font-size: 11px; text-transform: uppercase; font-weight: 800; letter-spacing: 1px; color: #9CA3AF;">{col}</div></div>'
        parts.append(f'<th style="text-align: center; color: {c_meta["color"]}; font-weight: bold; border-right: 1px solid #1F2937; width: 25%; vertical-align: middle;">{header_content}</th>')
    
    parts.append('</tr></thead><tbody>')
    
    for i in range(max_rows):
        bg_color = "#1F2937" if i % 2 == 0 else "#111827"
        parts.append(f'<tr style="background-color: {bg_color}; transition: background 0.2s;">')
        for col in required_cols:
            val = clean_data[col][i] if i < len(clean_data[col]) else ""
            text_color = meta[col]['color'] if val != "" else "transparent"
            font_w = "600" if val != "" else "normal"
            parts.append(f'<td style="padding: 10px; text-align: center; border-right: 1px solid #374151; color: {text_color}; font-weight: {font_w}; border-bottom: 1px solid #374151;">{val}</td>')
        parts.append("</tr>")
        
    parts.append("</tbody></table></div>")
    return "".join(parts)

def generate_board_html(grid_data, start_row, end_row, cell_classes, cell_styles=None):
    if cell_styles is None: cell_styles = {}
        
    html = '<div class="grid-container">'
    headers = [
        ('Spades', '♠', '#D1D5DB'),
        ('Hearts', '♥', '#EF4444'),
        ('Diamonds', '♦', '#EF4444'),
        ('Clubs', '♣', '#D1D5DB')
    ]
    for name, icon, color in headers:
        html += f'<div class="grid-header"><div class="suit-icon" style="color:{color};">{icon}</div><div class="suit-name" style="font-size: 11px; color: #9CA3AF; font-weight: 800; letter-spacing: 1px; text-transform: uppercase;">{name}</div></div>'
    
    for r in range(start_row, min(len(grid_data), end_row)):
        for c in range(4):
            val = str(grid_data[r, c])
            if val == 'nan' or not val: val = ''
            
            extra_classes = cell_classes.get((r, c), "")
            classes = "grid-cell"
            if extra_classes:
                classes += " " + extra_classes
                
            style = cell_styles.get((r, c), "")
                
            inner = val
            if "missing-marker" in classes or "missing-selected" in classes:
                inner = f'<div class="missing-circle">{val}</div>'
                classes = classes.replace("missing-marker", "").replace("missing-selected", "")
            
            html += f'<div class="{classes.strip()}" style="{style}">{inner}</div>'
                 
    html += '</div>'
    return html

def find_matches_for_pattern(shape_idx, selected_cards, grid_data, row_limit):
    found = []
    variations = generate_variations_strict(shape_idx, base_shapes[shape_idx])
    rows = min(len(grid_data), row_limit)
    colors = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444']
    
    raw_matches = []
    for shape in variations:
        sh_h = max(r for r, c in shape) + 1
        sh_w = max(c for r, c in shape) + 1
        
        for r in range(rows - sh_h + 1):
            for c in range(4 - sh_w + 1):
                vals = []
                coords = []
                try:
                    for dr, dc in shape:
                        vals.append(str(grid_data[r+dr, c+dc]))
                        coords.append((r+dr, c+dc))
                except IndexError:
                    continue
                
                used_indices = []
                temp_selected = selected_cards.copy()
                
                for i, v in enumerate(vals):
                    if v in temp_selected:
                        temp_selected.remove(v)
                        used_indices.append(i)
                
                if len(used_indices) == 3:
                    miss_i = [i for i in range(len(vals)) if i not in used_indices][0]
                    
                    if shape_idx == 1 and shape[miss_i][0] == 3:
                        continue
                        
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
    res = []
    for v in vals:
        if v != "-" and v not in seen:
            seen.add(v)
            res.append(v)
    return res

# --- Session State Initialization ---
if 'uploaded_df' not in st.session_state: st.session_state['uploaded_df'] = None
if 'current_shape_idx' not in st.session_state: st.session_state['current_shape_idx'] = 0
if 'window_start' not in st.session_state: st.session_state['window_start'] = 0
if 'sleep_window_start' not in st.session_state: st.session_state['sleep_window_start'] = 0
if 'num_combos_val' not in st.session_state: st.session_state['num_combos_val'] = 6

if 'chk_all' not in st.session_state:
    st.session_state['chk_all'] = False

for k in PATTERN_NAMES.keys():
    if f'chk_pat_{k}' not in st.session_state:
        st.session_state[f'chk_pat_{k}'] = False

def toggle_all():
    val = st.session_state.chk_all
    for k in PATTERN_NAMES.keys():
        st.session_state[f'chk_pat_{k}'] = val

base_shapes = parse_shapes_strict(FIXED_COMBOS_TXT)

st.title("⚡ Chance Analyzer PRO")

with st.sidebar:
    st.header("📂 Upload Data")
    csv_file = st.file_uploader("Choose a CSV file", type=None)
    st.markdown("---")
    
    st.header("⚙️ Algorithm Settings")
    search_depth = st.number_input("🔍 History Scan Depth", min_value=5, max_value=50000, value=26, step=1)

if csv_file:
    temp_df, msg = load_data_bulletproof(csv_file)
    if temp_df is not None:
        st.session_state['uploaded_df'] = temp_df

df = st.session_state['uploaded_df']

if df is not None:
    required_cols = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    df.columns = df.columns.str.strip()
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.stop()

    grid_data = df[required_cols].values
    ROW_LIMIT = 26
    max_val = max(0, len(grid_data) - 3)
    max_sleep_val = max(0, len(grid_data) - 1)

    def dec_win():
        st.session_state['window_start'] = max(0, st.session_state['window_start'] - 1)
    
    def inc_win():
        st.session_state['window_start'] = min(max_val, st.session_state['window_start'] + 1)
        
    def dec_sleep_win():
        st.session_state['sleep_window_start'] = max(0, st.session_state['sleep_window_start'] - 1)
        
    def inc_sleep_win():
        st.session_state['sleep_window_start'] = min(max_sleep_val, st.session_state['sleep_window_start'] + 1)
        
    def prev_pat():
        st.session_state['current_shape_idx'] = (st.session_state['current_shape_idx'] - 1) % len(PATTERN_NAMES)
        
    def next_pat():
        st.session_state['current_shape_idx'] = (st.session_state['current_shape_idx'] + 1) % len(PATTERN_NAMES)
        
    def reset_search():
        st.session_state['search_done'] = False

    tab_matches, tab_predictor, tab_sleep = st.tabs(["📋 PATTERN MATCHES", "🔍 3-ROW PREDICTOR", "💤 SLEEPING CARDS"])
    selected_match_ids = None 
    
    with tab_matches:
        with st.expander("⚙️ Configuration & Target Inputs", expanded=not st.session_state.get('search_done', False)):
            col_conf, col_prev = st.columns([4, 1])
            with col_conf:
                nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
                with nav_col1:
                    st.button("◀", use_container_width=True, key="prev_pat_btn", on_click=prev_pat)
                with nav_col2:
                    curr_name = PATTERN_NAMES[st.session_state['current_shape_idx']]
                    st.markdown(f"<div style='display: flex; align-items: center; justify-content: center; height: 2.8rem; background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; font-weight: 600; color: #58A6FF; font-size: 15px;'>{curr_name}</div>", unsafe_allow_html=True)
                with nav_col3:
                    st.button("▶", use_container_width=True, key="next_pat_btn", on_click=next_pat)
                        
                shape_idx = st.session_state['current_shape_idx']

            with col_prev:
                st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)
            
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            
            raw_cards = np.unique(grid_data.astype(str))
            clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])
            
            c1, c2, c3 = st.columns(3)
            with c1: card1 = st.selectbox("Card 1", [""] + clean_cards, key="c1", label_visibility="collapsed")
            with c2: card2 = st.selectbox("Card 2", [""] + clean_cards, key="c2", label_visibility="collapsed")
            with c3: card3 = st.selectbox("Card 3", [""] + clean_cards, key="c3", label_visibility="collapsed")
            
            selected_cards = [c for c in [card1, card2, card3] if c != ""]
            
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

            b_search, b_empty, b_reset = st.columns([3, 3, 1])
            with b_search: run_search = st.button("🔍 Search", type="primary", use_container_width=True)
            with b_reset: st.button("Reset", use_container_width=True, on_click=reset_search)

        found_matches = []
        
        if (run_search or st.session_state.get('search_done', False)) and len(selected_cards) == 3:
            st.session_state['search_done'] = True
            current_patt_idx = st.session_state.get('current_shape_idx', shape_idx)
            found_matches = find_matches_for_pattern(current_patt_idx, selected_cards, grid_data, search_depth)

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
            calc_height = (len(display_df) + 1) * 36 + 3
            
            event = st.dataframe(display_df.drop(columns=['Hidden_ID']), hide_index=True, use_container_width=True, selection_mode="single-row", on_select="rerun", height=calc_height)
            if len(event.selection['rows']) > 0:
                selected_match_ids = display_df.iloc[event.selection['rows'][0]]['Hidden_ID']
            
        st.markdown("<h4 style='margin-top: 15px; font-weight: 800; color: #F3F4F6;'>Live Game Board</h4>", unsafe_allow_html=True)
        cell_classes = {}
        matches_to_show = found_matches
        
        if selected_match_ids is not None:
            matches_to_show = [m for m in found_matches if m['id'] in selected_match_ids]

        for m in matches_to_show:
            col_idx = m['color_idx']
            for coord in m['full_coords_list']:
                if coord != m['miss_coords']:
                    if coord not in cell_classes: cell_classes[coord] = ""
                    cell_classes[coord] += f" match-color-{col_idx}"
                    
            miss = m['miss_coords']
            if miss not in cell_classes: cell_classes[miss] = ""
            if selected_match_ids is not None:
                cell_classes[miss] += " missing-selected"
            else:
                cell_classes[miss] += " missing-marker"
                
        draw_limit_matches = 30
        if selected_match_ids is not None and matches_to_show:
            max_r = max(coord[0] for m in matches_to_show for coord in m['full_coords_list'])
            draw_limit_matches = max(30, max_r + 3)
            
        st.markdown(generate_board_html(grid_data, 0, draw_limit_matches, cell_classes), unsafe_allow_html=True)

    with tab_predictor:
        c_minus, c_val, c_plus = st.columns([1, 2, 1])
        with c_minus:
            st.button("➖", use_container_width=True, key="btn_minus_pred", on_click=dec_win)
        with c_val:
            st.markdown(f"<div style='text-align:center; font-size: 18px; font-weight: 800; background: #1F2937; padding: 5px; border-radius: 8px; border: 1px solid #374151; color: #60A5FA;'>Row: {st.session_state['window_start']}</div>", unsafe_allow_html=True)
        with c_plus:
            st.button("➕", use_container_width=True, key="btn_plus_pred", on_click=inc_win)
                
        window_start = st.session_state['window_start']

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        with st.expander("⚙️ Pattern Filter Checklist", expanded=True):
            st.markdown("<div style='color: #9CA3AF; font-size: 13px; margin-bottom: 10px;'>Select patterns to include in the prediction calculation:</div>", unsafe_allow_html=True)
            
            st.checkbox("✅ Select / Deselect All", key="chk_all", on_change=toggle_all)
            st.markdown("<hr style='margin: 8px 0; border-color: #374151;'>", unsafe_allow_html=True)
            
            active_pattern_indices = []
            col1, col2, col3 = st.columns(3)
            items = list(PATTERN_NAMES.items())
            
            for i, (k, v) in enumerate(items):
                target_col = [col1, col2, col3][i % 3]
                if target_col.checkbox(v, key=f"chk_pat_{k}"):
                    active_pattern_indices.append(k)
                    
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        w_data = grid_data[window_start : window_start + 3, :]
        has_actual = window_start > 0
        actual_row = grid_data[window_start - 1, :] if has_actual else [None, None, None, None]
        
        historical_grid = grid_data[window_start : window_start + search_depth]
        
        if has_actual:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E3A8A 0%, #111827 100%); border: 1px solid #3B82F6; border-radius: 12px; padding: 15px; margin-top: 15px; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 10px rgba(59,130,246,0.2);">
                <div style="font-size: 13px; color: #93C5FD; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">🎯 Actual Outcome (Row {window_start - 1})</div>
                <div style="font-size: 22px; font-weight: 900; color: #FFFFFF; letter-spacing: 2px;">
                    <span style="color: #D1D5DB;">♠ {actual_row[0]}</span> &nbsp;|&nbsp; 
                    <span style="color: #EF4444;">♥ {actual_row[1]}</span> &nbsp;|&nbsp; 
                    <span style="color: #EF4444;">♦ {actual_row[2]}</span> &nbsp;|&nbsp; 
                    <span style="color: #D1D5DB;">♣ {actual_row[3]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #064E3B 0%, #111827 100%); border: 1px solid #10B981; border-radius: 12px; padding: 15px; margin-top: 15px; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 10px rgba(16,185,129,0.2);">
                <div style="font-size: 13px; color: #6EE7B7; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">🔮 Predicting Next Draw</div>
                <div style="font-size: 16px; font-weight: 600; color: #D1FAE5;">Waiting for actual results to validate...</div>
            </div>
            """, unsafe_allow_html=True)

        suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
        suit_icons = {'Spades': '♠', 'Hearts': '♥', 'Diamonds': '♦', 'Clubs': '♣'}
        suit_colors = {'Spades': '#D1D5DB', 'Hearts': '#EF4444', 'Diamonds': '#EF4444', 'Clubs': '#D1D5DB'}
        
        suit_tabs = st.tabs([f"{suit_icons[s]} {s}" for s in suits])
        suit_predictions = {}
        
        for i, suit in enumerate(suits):
            with suit_tabs[i]:
                triplet = [str(w_data[r, i]) for r in range(min(3, len(w_data))) if str(w_data[r, i]).lower() != 'nan' and str(w_data[r, i]).strip() != '']
                triplet_str = f"[ {' | '.join(triplet)} ]" if len(triplet) == 3 else "Incomplete"
                actual_card = str(actual_row[i]).strip().upper() if has_actual else "-"
                
                st.markdown(f"<div style='text-align: center; color: #8B949E; font-size: 13px; font-weight: 600; margin-bottom: 10px;'>Base Triplet: <span style='color: #D1D5DB;'>{triplet_str}</span></div>", unsafe_allow_html=True)
                
                html_table = "<table style='max-width: 400px; margin: 0 auto; width: 100%; border-collapse: collapse; text-align: center; background: #161B22; border-radius: 10px; overflow: hidden; border: 1px solid #30363D;'>"
                html_table += "<thead style='background: #21262D; color: #8B949E; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;'>"
                html_table += "<tr>"
                html_table += "<th style='padding: 8px; border-bottom: 1px solid #30363D; width: 60%;'>Predicted Card</th>"
                html_table += "<th style='padding: 8px; border-bottom: 1px solid #30363D; width: 40%;'>Matches</th>"
                html_table += "</tr></thead><tbody>"
                
                if len(triplet) == 3:
                    if not active_pattern_indices:
                        suit_predictions[suit] = []
                        html_table += "<tr style='border-bottom: 1px solid #30363D;'>"
                        html_table += "<td colspan='2' style='padding: 15px; color: #FCD34D; font-weight: 600;'>Select at least one pattern above to see predictions.</td>"
                        html_table += "</tr>"
                    else:
                        all_missing = []
                        for p_idx in active_pattern_indices:
                            m = find_matches_for_pattern(p_idx, triplet, historical_grid, search_depth)
                            all_missing.extend([x['miss_val'].strip().upper() for x in m])
                        
                        if all_missing:
                            counts = pd.Series(all_missing).value_counts()
                            suit_predictions[suit] = list(counts.index)
                            
                            for card, count in counts.items():
                                is_hit = has_actual and (card == actual_card)
                                
                                if is_hit:
                                    row_bg = "background: rgba(16, 185, 129, 0.1);"
                                    border_left = "border-left: 3px solid #10B981;"
                                    card_color = "#10B981"
                                    icon_check = "<span style='background: #10B981; color: white; padding: 2px 6px; border-radius: 4px; font-size: 9px; margin-left: 8px; vertical-align: middle; letter-spacing: 0.5px;'>HIT</span>"
                                else:
                                    row_bg = ""
                                    border_left = "border-left: 3px solid transparent;"
                                    card_color = "#58A6FF"
                                    icon_check = ""
                                
                                html_table += f"<tr style='{row_bg} border-bottom: 1px solid #30363D;'>"
                                html_table += f"<td style='padding: 8px; {border_left} font-weight: 900; font-size: 18px; color: {card_color}; border-bottom: 1px solid #30363D;'>{card}{icon_check}</td>"
                                html_table += f"<td style='padding: 8px; font-weight: 800; color: #FAFAFA; font-size: 16px; border-bottom: 1px solid #30363D;'>{count}</td>"
                                html_table += "</tr>"
                        else:
                            suit_predictions[suit] = []
                            html_table += "<tr style='border-bottom: 1px solid #30363D;'>"
                            html_table += "<td style='padding: 8px; border-left: 3px solid transparent; color: #F85149; font-weight: 600;'>No Match</td>"
                            html_table += "<td style='padding: 8px; color: #8B949E;'>0</td>"
                            html_table += "</tr>"
                else:
                    suit_predictions[suit] = []
                    html_table += "<tr style='border-bottom: 1px solid #30363D;'>"
                    html_table += "<td style='padding: 8px; border-left: 3px solid transparent; color: #8B949E; font-weight: 600;'>Incomplete Triplet</td>"
                    html_table += "<td style='padding: 8px; color: #8B949E;'>-</td>"
                    html_table += "</tr>"
                    
                html_table += "</tbody></table>"
                st.markdown(html_table, unsafe_allow_html=True)
                
        st.markdown("---")
        
        c_head, c_info = st.columns([2, 1])
        with c_head:
            st.markdown("<h3 style='margin: 0; color: #FAFAFA;'>🎲 Chance 3 Combinations (Safety Net)</h3>", unsafe_allow_html=True)
            num_combos = st.slider("Select Number of Tickets", min_value=1, max_value=10, key="num_combos_val", step=1, label_visibility="collapsed")
            
        def get_c(s, r):
            preds = suit_predictions.get(s, [])
            if len(preds) > r: return preds[r]
            if len(preds) > 0: return preds[0]
            return "-"
            
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
        
        selected_combos = combos[:num_combos]
        
        total_cost = 0
        if active_pattern_indices:
            for cb in selected_combos:
                ways = 1
                for vals in cb["cfg"]:
                    u_vals = get_unique_valid(vals)
                    if u_vals:
                        ways *= len(u_vals)
                total_cost += ways * 5

        with c_info:
            st.markdown(f"""
            <div style="background: #1F2937; border: 1px solid #374151; border-radius: 8px; padding: 10px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 12px; color: #9CA3AF; text-transform: uppercase; font-weight: 600;">Total Investment</div>
                <div style="font-size: 24px; color: #10B981; font-weight: 900;">₪{total_cost}</div>
            </div>
            """, unsafe_allow_html=True)

        if not active_pattern_indices:
            st.info("👆 Please select patterns from the checklist above to generate combinations.")
        else:
            html_combos = '<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; margin-top: 15px;">'
            for cb in selected_combos:
                html_combos += '<div style="flex: 1 1 300px; background: #1F2937; border: 1px solid #374151; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">'
                html_combos += '<div style="color: #FCD34D; font-weight: 800; font-size: 15px; border-bottom: 1px solid #374151; padding-bottom: 8px; margin-bottom: 10px; text-align: center;">' + cb["name"] + '</div>'
                html_combos += '<div style="display: flex; justify-content: space-around; align-items: center;">'
                for i, s in enumerate(suits):
                    u_vals = get_unique_valid(cb["cfg"][i])
                    val_str = " + ".join(u_vals) if u_vals else "-"
                    icon = suit_icons[s]
                    
                    if val_str == "-":
                        color = "#4B5563"
                        bg_style = "background: #111827; border: 1px dashed #374151; opacity: 0.4;"
                    else:
                        color = suit_colors[s]
                        is_chizuk = len(u_vals) > 1
                        bg_style = "background: rgba(59, 130, 246, 0.1); border: 1px dashed #3B82F6;" if is_chizuk else "background: #111827; border: 1px solid #374151;"
                    
                    html_combos += '<div style="text-align: center; padding: 8px; border-radius: 8px; ' + bg_style + ' width: 22%;">'
                    html_combos += '<div style="color: ' + color + '; font-size: 18px; margin-bottom: 4px;">' + icon + '</div>'
                    html_combos += '<div style="color: #FFF; font-weight: 900; font-size: 14px;">' + val_str + '</div>'
                    html_combos += '</div>'
                html_combos += '</div></div>'
            html_combos += '</div>'
            
            st.markdown(html_combos, unsafe_allow_html=True)
                
        st.markdown("<h4 style='margin-top: 25px; font-weight: 800; color: #FAFAFA;'>Historical Game Board</h4>", unsafe_allow_html=True)
        cell_classes_3row = {}
        draw_limit_pred = window_start + 30
        for r in range(max(0, window_start - 1), min(len(grid_data), draw_limit_pred)):
            for c in range(4):
                if window_start <= r < window_start + 3:
                    cell_classes_3row[(r, c)] = " window-highlight"
                else:
                    cell_classes_3row[(r, c)] = " window-dim"
                    
        st.markdown(generate_board_html(grid_data, max(0, window_start - 1), draw_limit_pred, cell_classes_3row), unsafe_allow_html=True)

    with tab_sleep:
        c_minus_s, c_val_s, c_plus_s = st.columns([1, 2, 1])
        with c_minus_s:
            st.button("➖", use_container_width=True, key="btn_minus_sleep", on_click=dec_sleep_win)
        with c_val_s:
            if st.session_state['sleep_window_start'] == 0:
                header_text = "Row: 0 (🔮 Predicting Next Draw)"
            else:
                header_text = f"Row: {st.session_state['sleep_window_start']} (🎯 Target: Row {st.session_state['sleep_window_start'] - 1})"
            st.markdown(f"<div style='text-align:center; font-size: 18px; font-weight: 800; background: #1F2937; padding: 5px; border-radius: 8px; border: 1px solid #374151; color: #10B981;'>{header_text}</div>", unsafe_allow_html=True)
        with c_plus_s:
            st.button("➕", use_container_width=True, key="btn_plus_sleep", on_click=inc_sleep_win)
            
        sleep_window_start = st.session_state['sleep_window_start']
        sleep_threshold = st.number_input("Sleep Threshold (Gap)", min_value=3, max_value=20, value=7, step=1)
            
        sleep_data_lists = {}
        active_sleepers = set()
        
        calc_start_idx = sleep_window_start
        
        for col_idx, col_name in enumerate(required_cols):
            if calc_start_idx >= len(grid_data):
                col_data = np.array([])
            else:
                col_data = grid_data[calc_start_idx:, col_idx]
                
            c_unique = np.unique(grid_data[:, col_idx].astype(str))
            lst = []
            for c in c_unique:
                if str(c).lower() == 'nan' or not str(c).strip(): continue
                locs = np.where(col_data == str(c))[0]
                gap = locs[0] if len(locs) > 0 else len(col_data)
                if gap >= sleep_threshold: 
                    lst.append((c, gap))
                    active_sleepers.add((col_idx, str(c)))
            lst.sort(key=lambda x: x[1], reverse=True)
            sleep_data_lists[col_name] = [f"{item[0]} : {item[1]}" for item in lst]

        if sleep_window_start == 0:
            st.markdown(f"<h4 style='margin-top: 15px; font-weight: 800; color: #F3F4F6;'>Sleeping Cards (Right Now, Heading into Next Draw)</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='margin-top: 15px; font-weight: 800; color: #F3F4F6;'>Sleeping Cards (Just Before Row {sleep_window_start - 1})</h4>", unsafe_allow_html=True)
        
        if any(sleep_data_lists.values()):
            st.markdown(create_sleeping_html_table(sleep_data_lists, required_cols), unsafe_allow_html=True)
        else:
            st.info("No sleeping cards found for the selected threshold.")
            
        cell_classes_sleep = {}
        
        if sleep_window_start == 0:
            st.markdown(f"<h4 style='margin-top: 25px; font-weight: 800; color: #F3F4F6;'>Current History Board</h4>", unsafe_allow_html=True)
            st.markdown("<div style='color: #9CA3AF; font-size: 13px; margin-bottom: 10px;'>Showing the most recent draws. No chain reactions exist yet because the next draw hasn't happened.</div>", unsafe_allow_html=True)
            
            for r in range(1, min(len(grid_data), 6)):
                for c in range(4):
                    cell_classes_sleep[(r, c)] = "window-dim"
                    
            st.markdown(generate_board_html(grid_data, 0, min(len(grid_data), 6), cell_classes_sleep), unsafe_allow_html=True)
            
        else:
            st.markdown(f"<h4 style='margin-top: 25px; font-weight: 800; color: #F3F4F6;'>Awakening Chain Reaction Tracker</h4>", unsafe_allow_html=True)
            st.markdown("<div style='color: #9CA3AF; font-size: 13px; margin-bottom: 10px;'>Green highlighted cells indicate sleeping cards the exact moment they woke up. The blue framed row is the target draw.</div>", unsafe_allow_html=True)
            
            target_draw = sleep_window_start - 1
            draw_limit_sleep = min(len(grid_data), target_draw + 4)
            
            for r in range(target_draw + 1, draw_limit_sleep):
                for c in range(4):
                    cell_classes_sleep[(r, c)] = "window-dim"
                    
            for r in range(calc_start_idx - 1, -1, -1):
                for c in range(4):
                    val = str(grid_data[r, c])
                    classes = ""
                    
                    if r == target_draw:
                        classes += " trigger-row"
                        
                    if (c, val) in active_sleepers:
                        classes += " awakened-card"
                        active_sleepers.remove((c, val)) 
                        
                    if classes:
                        cell_classes_sleep[(r, c)] = classes.strip()
                        
            st.markdown(generate_board_html(grid_data, 0, draw_limit_sleep, cell_classes_sleep), unsafe_allow_html=True)

else:
    st.info("👋 Upload a CSV file to get started.")
