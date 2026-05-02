import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Chance Analyzer PRO",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- KILL MOBILE KEYBOARD HACK ---
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

# ==========================================
# Fixed Patterns (A = Shape Block, X = Skip/Gap)
# ==========================================
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

# ==========================================
# Pattern Parsing & Variations Logic
# ==========================================
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

# ==========================================
# Premium CSS Styling
# ==========================================
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
    
    .missing-selected { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important; color: #000 !important; font-weight: 900 !important; border: 1px solid #FFF !important; box-shadow: 0 0 15px rgba(245, 158, 11, 0.8) !important; transform: scale(1.1); z-index: 100; }
    .missing-subtle { background-color: rgba(245, 158, 11, 0.15) !important; border: 1px dashed #F59E0B !important; color: #FCD34D !important; }
    
    .missing-circle { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: #FFFFFF; font-weight: 800; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(245, 158, 11, 0.7); margin: auto; border: 2px solid #FEF3C7; }
    
    .frame-box { position: absolute; top: 0; left: 0; right: 0; bottom: 0; border-style: solid; pointer-events: none; border-radius: 8px; z-index: 10; }
    
    .window-highlight { 
        border: 1px solid #F59E0B !important; 
        box-shadow: inset 0 0 15px rgba(245, 158, 11, 0.5), 0 0 8px rgba(245, 158, 11, 0.3) !important; 
        background-color: #1F2937 !important; 
        z-index: 5; 
    }
    .window-dim { opacity: 0.3 !important; filter: grayscale(40%); }

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
def load_data_robust(uploaded_file):
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
            
    hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
    df.rename(columns=hebrew_map, inplace=True)
    
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(r'[^0-9a-zA-Z]', '', regex=True)
        df[col] = df[col].replace('nan', '')
        
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

def generate_board_html(grid_data, start_row, end_row, cell_styles):
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
            
            style_extra = cell_styles.get((r, c), "")
            inner = val
            
            if "MISSING_SELECTED" in style_extra:
                style_extra = style_extra.replace("MISSING_SELECTED", "missing-selected")
            elif "MISSING_SUBTLE" in style_extra:
                style_extra = style_extra.replace("MISSING_SUBTLE", "missing-subtle")
                
            if "MISSING_MARKER" in style_extra:
                inner = f'<div class="missing-circle">{val}</div>'
                style_extra = style_extra.replace("MISSING_MARKER", "")
            
            if style_extra.strip().startswith("cell-") or style_extra.strip().startswith("window-"):
                 html += f'<div class="grid-cell {style_extra}">{inner}</div>'
            elif style_extra.strip() != "":
                 html += f'<div class="grid-cell {style_extra}">{inner}</div>'
            else:
                 html += f'<div class="grid-cell">{inner}</div>'
                 
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
        m['color'] = colors[i % len(colors)]
        found.append(m)
        
    return found

# ==========================================
# Main Interface
# ==========================================
st.title("⚡ Chance Analyzer PRO")

with st.sidebar:
    st.header("📂 Upload Data")
    csv_file = st.file_uploader("Choose a CSV file", type=None)
    st.markdown("---")
    
    st.header("⚙️ Algorithm Settings")
    search_depth = st.number_input("🔍 History Scan Depth", min_value=5, max_value=50000, value=26, step=1)
    st.caption("Determines how many rows backwards the algorithm scans for patterns.")

if 'uploaded_df' not in st.session_state: st.session_state['uploaded_df'] = None
if 'current_shape_idx' not in st.session_state: st.session_state['current_shape_idx'] = 0
if 'window_start' not in st.session_state: st.session_state['window_start'] = 0

base_shapes = parse_shapes_strict(FIXED_COMBOS_TXT)

if csv_file:
    temp_df, msg = load_data_robust(csv_file)
    if temp_df is not None:
        st.session_state['uploaded_df'] = temp_df
    elif msg != "ok":
        st.error(f"Error: {msg}")

df = st.session_state['uploaded_df']

if df is not None:
    required_cols = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    df.columns = df.columns.str.strip()
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        st.error(f"Missing columns in dataset: {missing}")
        st.stop()

    grid_data = df[required_cols].values
    ROW_LIMIT = 26
    
    tab_matches, tab_predictor, tab_sleep = st.tabs(["📋 PATTERN MATCHES", "🔍 3-ROW PREDICTOR", "💤 SLEEPING CARDS"])
    selected_match_ids = None 
    
    with tab_matches:
        with st.expander("⚙️ Configuration & Target Inputs", expanded=not st.session_state.get('search_done', False)):
            col_conf, col_prev = st.columns([4, 1])
            with col_conf:
                st.markdown("<label style='font-size: 14px; font-weight: 600; color: #FAFAFA;'>Search Pattern</label>", unsafe_allow_html=True)
                nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
                with nav_col1:
                    if st.button("◀", use_container_width=True, key="prev_pat"):
                        st.session_state['current_shape_idx'] = (st.session_state['current_shape_idx'] - 1) % len(PATTERN_NAMES)
                        st.rerun()
                with nav_col2:
                    curr_name = PATTERN_NAMES[st.session_state['current_shape_idx']]
                    st.markdown(f"<div style='display: flex; align-items: center; justify-content: center; height: 2.8rem; background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; font-weight: 600; color: #58A6FF; font-size: 15px;'>{curr_name}</div>", unsafe_allow_html=True)
                with nav_col3:
                    if st.button("▶", use_container_width=True, key="next_pat"):
                        st.session_state['current_shape_idx'] = (st.session_state['current_shape_idx'] + 1) % len(PATTERN_NAMES)
                        st.rerun()
                        
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
            with b_reset: reset_btn = st.button("Reset", use_container_width=True)
            
            if reset_btn:
                st.session_state['search_done'] = False
                st.rerun()

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
            
            st.markdown("<p style='color: #9CA3AF; font-size: 13px; font-weight: 600; margin-bottom: 5px;'>SELECT A ROW TO HIGHLIGHT ON BOARD:</p>", unsafe_allow_html=True)
            event = st.dataframe(display_df.drop(columns=['Hidden_ID']), hide_index=True, use_container_width=True, selection_mode="single-row", on_select="rerun", height=calc_height)
            if len(event.selection['rows']) > 0:
                selected_match_ids = display_df.iloc[event.selection['rows'][0]]['Hidden_ID']
        else:
            if st.session_state.get('search_done', False): 
                st.info("No matches found for the currently selected pattern.")
            
        st.markdown("<h4 style='margin-top: 15px; font-weight: 800; color: #F3F4F6;'>Live Game Board</h4>", unsafe_allow_html=True)
        cell_styles = {}
        matches_to_show = found_matches
        
        if selected_match_ids is not None:
            matches_to_show = [m for m in found_matches if m['id'] in selected_match_ids]

        for m in matches_to_show:
            col = m['color']
            for coord in m['full_coords_list']:
                if coord != m['miss_coords']:
                    if coord not in cell_styles: cell_styles[coord] = ""
                    count = cell_styles[coord].count("frame-box")
                    inset = count * 3
                    cell_styles[coord] += f'<div class="frame-box" style="border-width: 2px; border-color: {col}; box-shadow: inset 0 0 10px {col}, 0 0 8px {col}; top: {inset}px; left: {inset}px; right: {inset}px; bottom: {inset}px;"></div>'
            miss = m['miss_coords']
            if miss not in cell_styles: cell_styles[miss] = ""
            
            if selected_match_ids is not None:
                cell_styles[miss] += " MISSING_SELECTED"
            else:
                cell_styles[miss] += " MISSING_MARKER"
                
        draw_limit_matches = 30
        if selected_match_ids is not None and matches_to_show:
            max_r = max(coord[0] for m in matches_to_show for coord in m['full_coords_list'])
            draw_limit_matches = max(30, max_r + 3)
            
        st.markdown(generate_board_html(grid_data, 0, draw_limit_matches, cell_styles), unsafe_allow_html=True)

    with tab_predictor:
        st.markdown("<p style='color: #9CA3AF; font-size: 13px; font-weight: 600; margin-bottom: 5px;'>SELECT A BASE ROW FOR 3-ROW WINDOW:</p>", unsafe_allow_html=True)
        
        max_val = max(0, len(grid_data) - 3)
        
        c_minus, c_val, c_plus = st.columns([1, 2, 1])
        with c_minus:
            if st.button("➖", use_container_width=True, key="btn_minus"):
                st.session_state['window_start'] = max(0, st.session_state['window_start'] - 1)
                st.rerun()
        with c_val:
            st.markdown(f"<div style='text-align:center; font-size: 18px; font-weight: 800; background: #1F2937; padding: 5px; border-radius: 8px; border: 1px solid #374151; color: #60A5FA;'>Row: {st.session_state['window_start']}</div>", unsafe_allow_html=True)
        with c_plus:
            if st.button("➕", use_container_width=True, key="btn_plus"):
                st.session_state['window_start'] = min(max_val, st.session_state['window_start'] + 1)
                st.rerun()
                
        window_start = st.session_state['window_start']
        
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
        
        suit_tabs = st.tabs([f"{suit_icons[s]} {s}" for s in suits])
        
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
                    all_missing = []
                    for p_idx in range(len(base_shapes)):
                        m = find_matches_for_pattern(p_idx, triplet, historical_grid, search_depth)
                        all_missing.extend([x['miss_val'].strip().upper() for x in m])
                    
                    if all_missing:
                        counts = pd.Series(all_missing).value_counts()
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
                        html_table += "<tr style='border-bottom: 1px solid #30363D;'>"
                        html_table += "<td style='padding: 8px; border-left: 3px solid transparent; color: #F85149; font-weight: 600;'>No Match</td>"
                        html_table += "<td style='padding: 8px; color: #8B949E;'>0</td>"
                        html_table += "</tr>"
                else:
                    html_table += "<tr style='border-bottom: 1px solid #30363D;'>"
                    html_table += "<td style='padding: 8px; border-left: 3px solid transparent; color: #8B949E; font-weight: 600;'>Incomplete Triplet</td>"
                    html_table += "<td style='padding: 8px; color: #8B949E;'>-</td>"
                    html_table += "</tr>"
                    
                html_table += "</tbody></table>"
                st.markdown(html_table, unsafe_allow_html=True)
                
        st.markdown("<h4 style='margin-top: 25px; font-weight: 800; color: #FAFAFA;'>Historical Game Board</h4>", unsafe_allow_html=True)
        cell_styles_3row = {}
        draw_limit_pred = window_start + 30
        for r in range(max(0, window_start - 1), min(len(grid_data), draw_limit_pred)):
            for c in range(4):
                if window_start <= r < window_start + 3:
                    cell_styles_3row[(r, c)] = " window-highlight"
                else:
                    cell_styles_3row[(r, c)] = " window-dim"
                    
        st.markdown(generate_board_html(grid_data, max(0, window_start - 1), draw_limit_pred, cell_styles_3row), unsafe_allow_html=True)

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
            
        st.markdown("<h4 style='margin-top: 15px; font-weight: 800; color: #F3F4F6;'>Live Game Board</h4>", unsafe_allow_html=True)
        st.markdown(generate_board_html(grid_data, 0, ROW_LIMIT, {}), unsafe_allow_html=True)

else:
    st.info("👋 Upload a CSV file to get started.")
