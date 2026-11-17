"""Render the exact edited human artifact consumed by A's GMR, never dataset targets."""
import json,sys
from visualmimic.luna_target_pilot import ROOT, read, write
OUT=ROOT/'campaigns/luna_pregmr10_v1'
def main(stage):
 if stage=='prepare':
  OUT.mkdir(exist_ok=False)
  tasks=json.loads((ROOT/'campaigns/luna_abc10_v1/tasks.json').read_text())
  manifest=[]
  for row in tasks:
   result=json.loads(read(row['A']['result']))
   report=json.loads(read(result['retarget_report']))
   assert report['key']==row['key'] and result['instruction']==row['instruction']
   read(report['human'])
   row['pregmr_human']=report['human']
   manifest.append({'key':row['key'],'instruction':row['instruction'],'source':report['human']})
  assert len(tasks)==10
  write(OUT/'inputs.json',tasks);write(OUT/'human_manifest.json',manifest)
 elif stage=='finalize':
  tasks=json.loads((OUT/'inputs.json').read_text())
  videos={json.dumps(r['key'],sort_keys=True):r for r in json.loads((OUT/'render/manifest.json').read_text())}
  assert len(videos)==len(tasks)==10
  for row in tasks:
   v=videos[json.dumps(row['key'],sort_keys=True)]
   assert v['instruction']==row['instruction'];read(v['video'])
   row['candidates']={'pre_GMR':v}
  write(OUT/'tasks.json',tasks)
 else:raise ValueError(stage)
if __name__=='__main__':main(sys.argv[1])
