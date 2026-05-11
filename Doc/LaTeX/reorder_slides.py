"""
Script v2 - tach slide bang \begin{frame} va \end{frame} trực tiếp
"""
import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SRC = r"C:\Users\APC\Downloads\har-wisdm-bidirectional-lstm-rnns-stacked_lstm_wihout_BO\Doc\LaTeX\defense_slides_research.tex"

with open(SRC, encoding="utf-8") as f:
    raw = f.read()

# Tach preamble
doc_start = raw.index(r'\begin{document}') + len(r'\begin{document}')
preamble  = raw[:doc_start]

# Lay phan body, bo \end{document}
body = raw[doc_start:]
body = body.replace(r'\end{document}', '').rstrip()

# Tach tung khoi: moi khoi la % === comment + \begin{frame}...\end{frame}
# Dung regex de tach
pattern = re.compile(
    r'((?:[ \t]*\n)*'           # blank lines truoc
    r'(?:[ \t]*%[^\n]*\n)*'    # comment lines
    r'[ \t]*\\begin\{frame\}'  # \begin{frame}
    r'.*?'
    r'\\end\{frame\})',         # \end{frame}
    re.DOTALL
)

# Phan khong phai frame (section declarations, etc.)
# Ta se xu ly theo cach khac: split body by \begin{frame}

# Approach: chia body thanh cac segment
# - section lines
# - frame blocks

segments = []
pos = 0
for m in re.finditer(r'\\begin\{frame\}.*?\\end\{frame\}', body, re.DOTALL):
    # phan truoc frame nay (chua cac comment va section)
    before = body[pos:m.start()]
    segments.append(('pre', before))
    segments.append(('frame', m.group()))
    pos = m.end()
segments.append(('pre', body[pos:]))

# In ra de kiem tra
frames_only = [s for t,s in segments if t == 'frame']
print(f"Found {len(frames_only)} frames")
for i, f in enumerate(frames_only):
    title_m = re.search(r'\\begin\{frame\}\{(.+?)\}', f)
    title = title_m.group(1) if title_m else f.split('\n')[0][:60]
    print(f"  [{i:2d}] {title}")
