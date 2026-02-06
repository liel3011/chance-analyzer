import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Chance Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
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
# ==========================================

# --- CSS Styling (English / LTR) ---
st.markdown("""
<style>
    /* General LTR settings */
    .stApp { direction: ltr; text-align: left; background-color: #202020; color: #f0f0f0; }
    .stSelectbox, .stMultiSelect, .stButton, div[data-testid="stExpander"], div[data-testid="stSidebar"] { direction: ltr; text-align: left; }

    /* Visual Grid */
    .grid-container { 
        display: grid; 
        grid-template-columns: repeat(4, 1fr); 
        gap: 4px; 
        background-color: #1e1e1e; 
        padding: 5px; 
        border-radius: 8px; 
        margin-top: 10px; 
    }
    .grid-cell { 
        background-color: #333; 
        color: #eee; 
        padding: 0; 
        text-align: center; 
        border-radius: 4px; 
        font-family: 'Segoe UI', sans-serif; 
        font-size: 14px; 
        position: relative; 
        border: 1px solid #444; 
        height: 40px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
    }

    /* Missing Card - Bold Black on White */
    .missing-circle { 
        background-color: white; 
        color: black; 
        font-weight: bold; 
        border-radius: 50%; 
        width: 32px; 
        height: 32px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        box-shadow: 0 0 5px rgba(255,255,255,0.7); 
        z-index: 10; 
    }

    /* Colored Frames */
    .frame-box { 
        position: absolute; 
        top: 0; left: 0; right: 0; bottom: 0; 
        border-style: solid; 
        border-color: transparent; 
        pointer-events: none; 
    }

    /* Grid Headers */
    .grid-header { 
        text-align: center; 
        color: #aaa; 
        font-weight: bold; 
        font-size: 14px; 
        padding-bottom: 8px; 
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* Suit Icons */
    .suit-icon {
        font-size: 24px;
        line-height: 1;
        margin-bottom: 2px;
    }

    /* Shape Preview */
    .shape-preview-container { 
        display: grid; 
        gap: 2px; 
        background-color: #333; 
        padding: 5px; 
        border-radius: 4px; 
        width: fit-content; 
        margin-bottom: 10px; 
    }

    /* Fix for adjacent selectboxes */
    div[data-testid="column"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# --- Logic Functions ---

@st.cache_data
def load_data_robust(uploaded_file):
    if uploaded_file is None: return None, "No file uploaded"
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        # Map Hebrew headers to English if they exist
        hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
        df.rename(columns=hebrew_map, inplace=True)

        return df, "ok"
    except:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp1255')

            # Try mapping again with different encoding
            hebrew_map = {'תלתן': 'Clubs', 'יהלום': 'Diamonds', 'לב': 'Hearts', 'עלה': 'Spades'}
            df.rename(columns=hebrew_map, inplace=True)

            return df, "ok"
        except:
            return None, "Error loading file"


def parse_shapes_strict(text):
    """Parses text representation into coordinates."""
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
                    coords.append((r, c_idx));
                    c_idx += 1
                elif char == 'S':
                    c_idx += 1
                elif char == ' ':
                    # Smart space handling
                    prev = line[i - 1] if i > 0 else None
                    next_c = line[i + 1] if i < len(line) - 1 else None
                    if not (prev in ['A', 'S'] and next_c in ['A', 'S']): c_idx += 1
                i += 1
        if not coords: continue
        # Normalize (move to 0,0)
        min_c = min(c for r, c in coords)
        coords = [(r, c - min_c) for r, c in coords]
        shapes.append(coords)
    return shapes


def generate_variations_strict(shape_idx, base_shape):
    """
    Generates variations (rotations/reflections) based on shape index.
    """
    variations = set()

    # Pattern 1: Row only
    if shape_idx == 0:
        variations.add(tuple(sorted(base_shape)))

        # Pattern 2: Column only
    elif shape_idx == 1:
        variations.add(tuple(sorted(base_shape)))

    # Pattern 3: Diagonal
    elif shape_idx == 2:
        variations.add(tuple(sorted(base_shape)))  # Descending
        # Mirror (Ascending)
        max_c = max(c for r, c in base_shape)
        mirror = [(r, max_c - c) for r, c in base_shape]
        variations.add(tuple(sorted(mirror)))

    # Pattern 4: ZigZag
    elif shape_idx == 3:
        variations.add(tuple(sorted([(0, 0), (1, 1), (2, 2), (1, 2)])))  # Down Right
        variations.add(tuple(sorted([(0, 0), (1, 1), (2, 2), (1, 0)])))  # Down Left
        variations.add(tuple(sorted([(0, 2), (1, 1), (2, 0), (1, 2)])))  # Up Right
        variations.add(tuple(sorted([(0, 2), (1, 1), (2, 0), (1, 0)])))  # Up Left

    # Pattern 5: Bridge (Card underneath)
    elif shape_idx == 4:
        base = [(0, 0), (0, 1), (0, 3), (1, 1)]
        variations.add(tuple(sorted(base)))

        # Vertical Flip (Legs up)
        max_r = max(r for r, c in base)
        flipped = sorted([(max_r - r, c) for r, c in base])
        variations.add(tuple(flipped))

        # Mirrors for both
        for v in list(variations):
            w = max(c for r, c in v)
            mirror = [(r, w - c) for r, c in v]
            variations.add(tuple(sorted(mirror)))

    # Patterns 7+: Row/Linear based (Horizontal + Mirrors + Flips)
    else:
        variations.add(tuple(sorted(base_shape)))

        # Horizontal Mirror
        w = max(c for r, c in base_shape)
        mirror_h = sorted([(r, w - c) for r, c in base_shape])
        variations.add(tuple(mirror_h))

        # Vertical Flip
        max_r = max(r for r, c in base_shape)
        flip_v = sorted([(max_r - r, c) for r, c in base_shape])
        variations.add(tuple(flip_v))

        # Flip + Mirror
        flip_hv = sorted([(max_r - r, w - c) for r, c in base_shape])
        variations.add(tuple(flip_hv))

    return [list(v) for v in variations]


def draw_preview_html(shape_coords):
    if not shape_coords: return ""
    min_r = min(r for r, c in shape_coords);
    min_c = min(c for r, c in shape_coords)
    norm = [(r - min_r, c - min_c) for r, c in shape_coords]
    max_r = max(r for r, c in norm) + 1;
    max_c = max(c for r, c in norm) + 1

    # Using no indentation in the inner HTML string to avoid markdown code block issues
    grid_html = f'<div class="shape-preview-container" style="display:grid; grid-template-columns: repeat({max_c}, 20px);">'
    for r in range(max_r):
        for c in range(max_c):
            bg = "#007acc" if (r, c) in norm else "#444"
            grid_html += f'<div style="width:20px; height:20px; border-radius:2px; background-color:{bg};"></div>'
    grid_html += '</div>'
    return grid_html


# --- Main Interface ---

st.title("📱 Chance Analyzer")

with st.sidebar:
    st.header("📂 Upload Data")
    csv_file = st.file_uploader("Upload CSV", type=['csv'])

df = None
base_shapes = parse_shapes_strict(FIXED_COMBOS_TXT)

if csv_file:
    df, msg = load_data_robust(csv_file)
    if df is None: st.error(f"Error: {msg}")

if df is not None:
    # Standardize Column Names
    required_cols = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
    df.columns = df.columns.str.strip()

    # Check for missing columns
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    grid_data = df[required_cols].values
    ROW_LIMIT = 51

    col_ctrl, col_visual = st.columns([1, 2])

    with col_ctrl:
        st.subheader("⚙️ Settings")

        # Shape Selection
        shape_idx = st.selectbox("Select Pattern:", range(len(base_shapes)), format_func=lambda x: f"Pattern {x + 1}")
        st.markdown("**Preview:**")
        st.markdown(draw_preview_html(base_shapes[shape_idx]), unsafe_allow_html=True)

        st.write("---")

        # Card Selection (3 separate boxes)
        raw_cards = np.unique(grid_data.astype(str))
        clean_cards = sorted([c for c in raw_cards if str(c).lower() != 'nan' and str(c).strip() != ''])

        st.markdown("**Select 3 Cards:**")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            c1 = st.selectbox("1", [""] + clean_cards, key="c1")
        with sc2:
            c2 = st.selectbox("2", [""] + clean_cards, key="c2")
        with sc3:
            c3 = st.selectbox("3", [""] + clean_cards, key="c3")

        selected_cards = [c for c in [c1, c2, c3] if c != ""]

        st.write("")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            run_search = st.button("🔍 Search", type="primary", use_container_width=True)
        with col_b2:
            reset_btn = st.button("Reset", use_container_width=True)

        if reset_btn:
            st.session_state['search_done'] = False
            st.session_state['selected_match'] = None

    # Search Logic
    found_matches = []
    if (run_search or st.session_state.get('search_done', False)) and len(selected_cards) == 3:
        st.session_state['search_done'] = True

        variations = generate_variations_strict(shape_idx, base_shapes[shape_idx])
        rows = min(len(grid_data), ROW_LIMIT)
        colors = ['#00ff99', '#ffcc00', '#ff66cc', '#00ccff', '#ff5050', '#cc99ff', '#ffff00']

        raw_matches = []
        for shape in variations:
            sh_h = max(r for r, c in shape) + 1;
            sh_w = max(c for r, c in shape) + 1
            for r in range(rows - sh_h + 1):
                for c in range(4 - sh_w + 1):
                    vals = [];
                    coords = []
                    try:
                        for dr, dc in shape:
                            vals.append(grid_data[r + dr, c + dc])
                            coords.append((r + dr, c + dc))
                    except:
                        continue

                    matched = 0;
                    used = set()
                    # Logic supporting duplicates
                    for t in selected_cards:
                        for i, v in enumerate(vals):
                            if i not in used and str(v) == t:
                                used.add(i);
                                matched += 1;
                                break

                    if matched == 3:
                        miss_i = [i for i in range(4) if i not in used][0]
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
            m['id'] = i + 1;
            m['color'] = colors[i % len(colors)]
            found_matches.append(m)

    # Tables Area
    with col_ctrl:
        st.write("---")
        tab1, tab2 = st.tabs(["📋 Results", "💤 Sleeping"])
        selected_match_id = None

        with tab1:
            if found_matches:
                df_res = pd.DataFrame(
                    [{'ID': m['id'], 'Missing': m['miss_val'], 'Row': m['miss_coords'][0]} for m in found_matches])
                event = st.dataframe(df_res, hide_index=True, use_container_width=True, selection_mode="single-row",
                                     on_select="rerun", height=250)
                if len(event.selection['rows']) > 0:
                    selected_match_id = df_res.iloc[event.selection['rows'][0]]['ID']
            else:
                if st.session_state.get('search_done', False): st.warning("No matches found")

        with tab2:
            sleep_txt = ""
            for i, col_name in enumerate(required_cols):
                sleep_txt += f"**{col_name}**\n"
                col_data = grid_data[:, i]
                c_unique = np.unique(col_data.astype(str))
                lst = []
                for c in c_unique:
                    if str(c).lower() == 'nan': continue
                    locs = np.where(col_data == c)[0]
                    if len(locs) > 0 and locs[0] > 7: lst.append((c, locs[0]))
                lst.sort(key=lambda x: x[1], reverse=True)
                if lst:
                    for c, g in lst: sleep_txt += f"- {c}: {g}\n"
                else:
                    sleep_txt += "- None\n"
                sleep_txt += "\n"
            st.text(sleep_txt)

    # Visual Grid
    with col_visual:
        st.subheader("📊 Game Board")
        cell_styles = {}
        matches_to_show = found_matches
        if selected_match_id is not None:
            matches_to_show = [m for m in found_matches if m['id'] == selected_match_id]

        for m in matches_to_show:
            col = m['color']
            for coord in m['full_coords_list']:
                if coord != m['miss_coords']:
                    if coord not in cell_styles: cell_styles[coord] = ""
                    count = cell_styles[coord].count("frame-box");
                    inset = count * 3
                    cell_styles[
                        coord] += f'<div class="frame-box" style="border-width: 2px; border-color: {col}; top: {inset}px; left: {inset}px; right: {inset}px; bottom: {inset}px;"></div>'

            miss = m['miss_coords']
            if miss not in cell_styles: cell_styles[miss] = ""
            cell_styles[miss] += "MISSING_MARKER"

        html = '<div class="grid-container">'

        # Headers with Icons (No Indentation to avoid markdown code block issues)
        headers = [('Clubs', '♣', '#e0e0e0'), ('Diamonds', '♦', '#ff4d4d'), ('Hearts', '♥', '#ff4d4d'),
                   ('Spades', '♠', '#e0e0e0')]

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