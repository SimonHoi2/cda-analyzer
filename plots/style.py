# ---------------------------------------------------------------------------
# Colour scheme (identiek aan desktop versie)
# ---------------------------------------------------------------------------
DARK_BG   = '#1e1e1e'
PANEL_BG  = '#2b2b2b'
OUT_CLR   = '#00e676'
BACK_CLR  = '#ff9100'
TURN_CLR  = '#f44336'
SPEED_CLR = '#00bcd4'
POWER_CLR = '#ffa726'
VE_CLR    = '#69f0ae'

PAIR_COLOURS = [
    ('#00e676', '#69f0ae'),
    ('#ff9100', '#ffcc02'),
    ('#b39ddb', '#e040fb'),
    ('#80cbc4', '#00bcd4'),
    ('#ef9a9a', '#f44336'),
    ('#a5d6a7', '#388e3c'),
]

LEG_PALETTE = [OUT_CLR, BACK_CLR, '#b39ddb', '#80cbc4', '#ffcc80', '#ef9a9a']

def style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    ax.title.set_color('white')
    for sp in ax.spines.values():
        sp.set_color('#444444')