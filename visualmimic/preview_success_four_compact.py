"""Read-only, loopback-only preview of three existing paired evaluation videos."""
import hashlib,json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from html import escape
ROOT=Path('/storage/ice1/3/5/szhang957/Repos/VisualMimic-20261025')
selected=['001396','002370','003994']
tasks={}
for campaign in ['luna_extend30_v1','luna_extend100_v1']:
 tasks.update({x['key']['sample_id']:x for x in json.loads((ROOT/'campaigns'/campaign/'tasks.json').read_text())})
routes={};cards=[]
for sid in selected:
 row=tasks[sid];videos={"source":row['source_video'],"human":row['candidates']['pre_GMR'],"robot":row['candidates']['C'],"A":row['candidates']['A'],"B":row['candidates']['B']}
 for name,v in videos.items():
  ref=v['video'];p=Path(ref['uri']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'];routes[f'/video/{sid}/{name}.mp4']=p
 pair=''.join(f'<div><h3>{label}</h3><video id="{sid}-{name}" controls preload="metadata" src="/video/{sid}/{name}.mp4"></video></div>' for name,label in [('human','① 编辑后人体 · GMR 前'),('robot','② C · GMR 参考轨迹，未执行'),('A','③ A · 人体输入，实际执行'),('B','④ B · 视频输入，实际执行')])
 cards.append(f'''<section><h2>{sid} · {escape(row['instruction'])}</h2><p>该样本四个阶段均被 Luna 判为成功。这是自动判定，请结合原指令和源动作自行核对。</p><button onclick="playPair('{sid}')">四段从头同步播放</button><div class="pair">{pair}</div><details><summary>查看原始源动作</summary><video controls preload="none" src="/video/{sid}/source.mp4"></video></details></section>''')
html=('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>四阶段成功样本视频对照</title><style>body{font:16px system-ui;margin:24px auto;padding:0 20px;max-width:1300px;background:#f6f7f9;color:#17212b}section{background:white;border:1px solid #dfe3e8;border-radius:12px;padding:12px;margin:12px 0}h2{font-size:17px;margin:8px 0}h3{font-size:13px;margin:6px 0}h1{font-size:22px;margin:10px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#eee;border-radius:8px}details video{max-width:640px}button{padding:8px 16px;margin:8px 0;cursor:pointer}p{color:#56616d}summary{cursor:pointer}@media(max-width:700px){body{margin:10px auto;padding:0 10px}section{padding:12px}.pair{gap:10px}h3{font-size:12px}button{padding:7px}h1{font-size:22px}}</style><h1>四阶段成功样本视频对照</h1><p>三个四阶段均判成功的样本，按原始速度同步播放。① 编辑后人体 → ② C 参考轨迹 → ③ A 实际执行；④ B 为视频输入分支。</p>'''+''.join(cards)+'''<script>
document.querySelectorAll('video').forEach(v=>{
 const c=document.createElement('canvas');c.width=640;c.height=480;c.style.cssText='width:100%;height:23vh;object-fit:contain;display:block;background:#eee;border-radius:8px';v.before(c);
 const controls=document.createElement('div');const btn=document.createElement('button');btn.textContent='播放 / 暂停';btn.onclick=()=>v.paused?v.play():v.pause();
 const seek=document.createElement('input');seek.type='range';seek.min=0;seek.max=4;seek.step=.01;seek.value=0;seek.style.width='55%';seek.oninput=()=>{v.currentTime=Number(seek.value)};
 const time=document.createElement('span');if(v.id)controls.style.display='none';controls.append(btn,seek,time);v.before(controls);v.style.display='none';v.preload='auto';v.muted=true;v.playsInline=true;
 const ctx=c.getContext('2d');function draw(){if(v.readyState>=2){ctx.drawImage(v,0,0,640,480);seek.max=v.duration;seek.value=v.currentTime;time.textContent=' '+v.currentTime.toFixed(2)+' / '+v.duration.toFixed(2)+'s';}requestAnimationFrame(draw)}draw();
});
function playPair(id){document.querySelectorAll('video').forEach(v=>v.pause());for(const name of ['human','robot','A','B']){const v=document.getElementById(id+'-'+name);v.currentTime=0;v.play();}}</script></html>''').encode()
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
print('Verified 15 source/four-stage videos. Serving on ICE loopback port 18772.',flush=True)
ThreadingHTTPServer(('127.0.0.1',18772),Handler).serve_forever()
