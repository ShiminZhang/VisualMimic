"""Read-only, loopback-only preview of three existing paired evaluation videos."""
import hashlib,json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from html import escape
ROOT=Path('/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025')
selected=['005139','004954','003282']
tasks={x['key']['sample_id']:x for x in json.loads((ROOT/'campaigns/luna_extend30_v1/tasks.json').read_text())}
routes={};cards=[]
for sid in selected:
 row=tasks[sid];videos={"source":row['source_video'],"human":row['candidates']['pre_GMR'],"robot":row['candidates']['C']}
 for name,v in videos.items():
  ref=v['video'];p=Path(ref['uri']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'];routes[f'/video/{sid}/{name}.mp4']=p
 pair=''.join(f'<div><h3>{label}</h3><video id="{sid}-{name}" controls preload="metadata" src="/video/{sid}/{name}.mp4"></video></div>' for name,label in [('human','Left: edited human motion before GMR'),('robot','Right: G1 reference after GMR')])
 cards.append(f'''<section><h2>{sid} · {escape(row['instruction'])}</h2><p>For all three pairs, Luna marked the human edit successful and the G1 reference unsuccessful. Check these judgments against the motion.</p><div class="pair">{pair}</div><button onclick="playPair('{sid}')">Play both from the start</button><details><summary>Show source motion</summary><video controls preload="none" src="/video/{sid}/source.mp4"></video></details></section>''')
html=('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Motion before and after GMR</title><style>body{font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1300px;background:#f6f7f9;color:#17212b}section{background:white;border:1px solid #dfe3e8;border-radius:12px;padding:20px;margin:24px 0}h2{font-size:19px}h3{font-size:15px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#eee;border-radius:8px}details video{max-width:640px}button{padding:10px 18px;margin:16px 0;cursor:pointer}p{color:#56616d}summary{cursor:pointer}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><h1>Motion before and after GMR</h1><p>Three selected diagnostic cases. Each clip plays at its original speed and frame rate. Expand a case to view its source motion.</p>'''+''.join(cards)+'''<script>function playPair(id){document.querySelectorAll('video').forEach(v=>v.pause());for(const name of ['human','robot']){const v=document.getElementById(id+'-'+name);v.currentTime=0;v.play();}}</script></html>''').encode()
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  path=self.path.split('?')[0]
  if path=='/':
   self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(html)));self.end_headers();self.wfile.write(html);return
  if path not in routes:self.send_error(404);return
  p=routes[path];size=p.stat().st_size;start,end=0,size-1;partial=False
  rng=self.headers.get('Range')
  if rng:
   import re
   m=re.fullmatch(r'bytes=(\d+)-(\d*)',rng)
   if not m:self.send_error(416);return
   start=int(m[1]);end=min(int(m[2]) if m[2] else end,end);partial=True
   if start>end:self.send_error(416);return
  self.send_response(206 if partial else 200);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
  if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  try:
   with p.open('rb') as f:
    f.seek(start);remaining=end-start+1
    while remaining:
     b=f.read(min(remaining,65536))
     if not b:break
     self.wfile.write(b);remaining-=len(b)
  except (BrokenPipeError,ConnectionResetError):pass
print('Verified 9 source/pair videos. Serving on ICE loopback port 18769.',flush=True)
ThreadingHTTPServer(('127.0.0.1',18769),Handler).serve_forever()
