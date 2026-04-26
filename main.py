import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

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
# Fixed Patterns
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

PLUS_SET = {"8", "10", "Q", "A"}
MINUS_SET = {"7", "9", "J", "K"}

def get_card_sign(card_val):
    val = str(card_val).strip().upper()
    if val in PLUS_SET: return "+"
    if val in MINUS_SET: return "-"
    return "?"

def analyze_pair_gap(df, col1, col2):
    s1 = df[col1].apply(get_card_sign)
    s2 = df[col2].apply(get_card_sign)
    pairs_series = s1 + s2 
    target_pairs = ["++", "--", "+-", "-+"]
    results = []
    for p in target_pairs:
        matches = (pairs_series == p)
        if matches.any():
            last_idx = matches.idxmax()
            results.append({'pair': p, 'ago': last_idx})
        else:
            results.append({'pair': p, 'ago': 9999})
    results.sort(key=lambda x: x['ago'], reverse=True)
    return results

def parse_shapes_strict(text):
    shapes = []
    blocks = text.replace('\r\n', '\n').split('\n\n')
    for block in blocks:
        if not block.strip(): continue
        lines = block.strip().split('\n')
        coords = []
        for r, line in enumerate(lines):
            chars = line.replace(' ', '')
            for c, char in enumerate(chars):
                if char == 'A': coords.append((r, c))
        if not coords: continue
        min_r, min_c = min(r for r, c in coords), min(c for r, c in coords)
        normalized = [(r - min_r, c - min_c) for r, c in coords]
        shapes.append(normalized)
    return shapes

def normalize_shape(shape):
    if not shape: return tuple()
    min_r, min_c = min(r for r, c in shape), min(c for r, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))

def generate_variations_strict(shape_idx, base_shape):
    bases = [normalize_shape(base_shape)]
    if shape_idx == 3: bases.append(normalize_shape([(0,0), (0,1), (1,1), (3,3)]))
    if shape_idx == 11: bases.append(normalize_shape([(0,0), (0,1), (1,1), (2,2)]))
    
    variations = set()
    for b in bases:
        variations.add(b)
        w, h = max(c for r, c in b), max(r for r, c in b)
        mirror_h = normalize_shape([(r, w - c) for r, c in b])
        flip_v = normalize_shape([(h - r, c) for r, c in b])
        rot_180 = normalize_shape([(h - r, w - c) for r, c in b])
        
        if shape_idx in [0, 1]: pass 
        elif shape_idx in [9, 10, 11]: variations.update([flip_v])
        elif shape_idx == 12: variations.update([mirror_h])
        elif shape_idx == 3: variations.update([mirror_h, flip_v, rot_180])
        else:
            rot_90 = normalize_shape([(c, h - r) for r, c in b])
            rot_270 = normalize_shape([(w - c, r) for r, c in b])
            transp1 = normalize_shape([(c, r) for r, c in b]) 
            transp2 = normalize_shape([(w - c, h - r) for r, c in b]) 
            variations.update([mirror_h, flip_v, rot_180, rot_90, rot_270, transp1, transp2])
            
    return [list(v) for v in variations if max(c for r, c in v) < 4]

# ==========================================
# Premium CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { direction: ltr; text-align: left; background-color: #0B0F19; color: #F3F4F6; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; }
    div[data-baseweb="select"] > div { background-color: #111827; border: 1px solid #1F2937; border-radius: 8px; }
    div.stButton > button { width: 100%; border-radius: 8px; height: 2.8rem; font-weight: 600; border: 1px solid #374151; background: #1F2937; color: #F9FAFB; }
    div.stButton > button[kind="primary"] { background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); border: none; }
    .grid-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; background: #111827; padding: 12px; border-radius: 16px; border: 1px solid #1F2937; }
    .grid-cell { background-color: #1F2937; color: #D1D5DB; border-radius: 8px; height: 42px; display: flex; align-items: center; justify-content: center; font-weight: 600; position: relative; border: 1px solid #374151; }
    .cell-plus { color: #10B981 !important; font-weight: 800 !important; } 
    .cell-minus { color: #F43F5E !important; font-weight: 800 !important; } 
    .missing-circle { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: #FFFFFF; font-weight: 800; border-radius: 50%; width: 30px; height: 32px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(245, 158, 11, 0.7); border: 2px solid #FEF3C7; }
    .frame-box { position: absolute; top: 0; left: 0; right: 0; bottom: 0; border-style: solid; pointer-events: none; border-radius: 8px; z-index: 10; }
    .winner-banner { background: linear-gradient(135deg, #065F46 0%, #047857 100%); border: 1px solid #10B981; border-radius: 12px; padding: 16px; text-align: center; color: #ECFDF5; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_robust(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='cp1255')
    df.rename(columns={'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}, inplace=True)
    return df

def generate_board_html(grid_data, row_limit, cell_styles):
    html = '<div class="grid-container">'
    headers = [('Spades', '♠', '#D1D5DB'), ('Hearts', '♥', '#EF4444'), ('Diamonds', '♦', '#EF4444'), ('Clubs', '♣', '#D1D5DB')]
    for name, icon, color in headers:
        html += f'<div style="text-align:center;"><div style="color:{color}; font-size:24px;">{icon}</div><div style="font-size:10px; color:#9CA3AF;">{name}</div></div>'
    for r in range(min(len(grid_data), row_limit)):
        for c in range(4):
            val = str(grid_data[r, c]) if str(grid_data[r, c]) != 'nan' else ''
            style = cell_styles.get((r, c), "")
            inner = f'<div class="missing-circle">{val}</div>' if "MISSING_MARKER" in style else val
            clean_style = style.replace("MISSING_MARKER", "")
            html += f'<div class="grid-cell">{inner}{clean_style}</div>'
    return html + '</div>'

def find_matches_for_pattern(shape_idx, selected_cards, grid_data, row_limit, base_shapes):
    found = []
    variations = generate_variations_strict(shape_idx, base_shapes[shape_idx])
    colors = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444']
    raw_matches = []
    for shape in variations:
        sh_h, sh_w = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
        for r in range(min(len(grid_data), row_limit) - sh_h + 1):
            for c in range(4 - sh_w + 1):
                vals = [str(grid_data[r+dr, c+dc]) for dr, dc in shape]
                coords = [(r+dr, c+dc) for dr, dc in shape]
                temp_sel = selected_cards.copy()
                used = [i for i, v in enumerate(vals) if v in temp_sel and not temp_sel.remove(v)]
                if len(used) == 3:
                    miss_i = [i for i in range(len(vals)) if i not in used][0]
                    if shape_idx == 1 and shape[miss_i][0] == 3: continue
                    m_data = {'coords': tuple(sorted(coords)), 'miss_coords': coords[miss_i], 'miss_val': vals[miss_i], 'full_coords_list': coords}
                    if not any(x['coords'] == m_data['coords'] for x in raw_matches): raw_matches.append(m_data)
    for i, m in enumerate(sorted(raw_matches, key=lambda x: x['miss_coords'][0])):
        m['id'], m['color'] = i + 1, colors[i % len(colors)]
        found.append(m)
    return found

# ==========================================
# Main
# ==========================================
st.title("⚡ Chance Analyzer PRO")
with st.sidebar:
    csv_file = st.file_uploader("📂 Upload CSV", type=None)

if 'uploaded_df' not in st.session_state: st.session_state['uploaded_df'] = None
if 'current_shape_idx' not in st.session_state: st.session_state['current_shape_idx'] = 0

base_shapes = parse_shapes_strict(FIXED_COMBOS_TXT)
if csv_file: st.session_state['uploaded_df'] = load_data_robust(csv_file)
df = st.session_state['uploaded_df']

if df is not None:
    grid_data, ROW_LIMIT = df[['Spades', 'Hearts', 'Diamonds', 'Clubs']].values, 26
    
    with st.expander("⚙️ Settings", expanded=not st.session_state.get('search_done', False)):
        c1, c2, c3, c4 = st.columns([1, 4, 1, 2])
        with c1:
            if st.button("◀"): st.session_state['current_shape_idx'] = (st.session_state['current_shape_idx'] - 1) % len(PATTERN_NAMES); st.rerun()
        with c2:
            st.markdown(f"<div style='display:flex;align-items:center;justify-content:center;height:2.8rem;background:#1F2937;border-radius:8px;font-weight:800;color:#60A5FA;'>{PATTERN_NAMES[st.session_state['current_shape_idx']]}</div>", unsafe_allow_html=True)
        with c3:
            if st.button("▶"): st.session_state['current_shape_idx'] = (st.session_state['current_shape_idx'] + 1) % len(PATTERN_NAMES); st.rerun()
        with c4:
            st.markdown(draw_preview_html(base_shapes[st.session_state['current_shape_idx']]), unsafe_allow_html=True)
        
        cards = sorted([c for c in np.unique(grid_data.astype(str)) if str(c).lower() != 'nan' and str(c).strip() != ''])
        i1, i2, i3 = st.columns(3)
        with i1: card1 = st.selectbox("C1", [""] + cards, key="c1", label_visibility="collapsed")
        with i2: card2 = st.selectbox("C2", [""] + cards, key="c2", label_visibility="collapsed")
        with i3: card3 = st.selectbox("C3", [""] + cards, key="c3", label_visibility="collapsed")
        
        sel = [c for c in [card1, card2, card3] if c != ""]
        b1, b2, b3 = st.columns([3, 3, 1])
        with b1: run_search = st.button("🔍 Search", type="primary")
        with b2: run_best = st.button("🏆 Winning Pattern", type="primary")
        with b3: 
            if st.button("Reset"): st.session_state['search_done'] = False; st.session_state.pop('winning_msg', None); st.rerun()

    found = []
    if run_best and len(sel) == 3:
        best_c, best_idx = -1, 0
        for i in range(len(base_shapes)):
            m = find_matches_for_pattern(i, sel, grid_data, ROW_LIMIT, base_shapes)
            if len(m) > best_c: best_c, best_idx = len(m), i
        st.session_state.update({'current_shape_idx': best_idx, 'winning_msg': f"🏆 Winning: {PATTERN_NAMES[best_idx]} ({best_c} matches)", 'search_done': True}); st.rerun()

    if (run_search or st.session_state.get('search_done', False)) and len(sel) == 3:
        st.session_state['search_done'] = True
        if 'winning_msg' in st.session_state: st.success(st.session_state['winning_msg'])
        found = find_matches_for_pattern(st.session_state['current_shape_idx'], sel, grid_data, ROW_LIMIT, base_shapes)

    t1, t2, t3 = st.tabs(["📋 MATCHES", "💤 SLEEPING", "⚖️ PAIRS"])
    with t1:
        selected_ids = None
        if found:
            raw = pd.DataFrame([{'Missing': m['miss_val'], 'Row': m['miss_coords'][0], 'id': m['id']} for m in found])
            gp = raw.groupby('Missing').agg({'Row': list, 'id': list}).reset_index()
            gp['Count'] = gp['id'].apply(len)
            gp = gp.sort_values('Count', ascending=False)
            ev = st.dataframe(gp[['Missing', 'Count']], hide_index=True, use_container_width=True, selection_mode="single-row", on_select="rerun")
            if ev.selection['rows']: selected_ids = gp.iloc[ev.selection['rows'][0]]['id']
        
        styles = {}
        to_show = found if not selected_ids else [m for m in found if m['id'] in selected_ids]
        for m in to_show:
            for coord in m['full_coords_list']:
                if coord != m['miss_coords']:
                    if coord not in styles: styles[coord] = ""
                    ins = styles[coord].count("frame-box") * 3
                    styles[coord] += f'<div class="frame-box" style="border-width:2px; border-color:{m["color"]}; box-shadow:0 0 8px {m["color"]}; top:{ins}px; left:{ins}px; right:{ins}px; bottom:{ins}px;"></div>'
            styles[m['miss_coords']] = styles.get(m['miss_coords'], "") + "MISSING_MARKER"
        st.markdown(generate_board_html(grid_data, ROW_LIMIT, styles), unsafe_allow_html=True)
else:
    st.info("👋 Upload CSV to start.")
