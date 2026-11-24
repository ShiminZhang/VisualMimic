"""Append fixed cohort samples 31-100 to the thirty-sample paired Luna pilot."""
import json,sys
from collections import Counter
from visualmimic.luna_target_pilot import ROOT,read,write
from visualmimic.render_provenance import verify_pre_gmr_videos
from visualmimic.judgment_accounting import combine_judgments
OUT=ROOT/'campaigns/luna_extend100_v1'
GROUPS=('pre_GMR','A','B','C')
def main(stage):
 if stage=='prepare':
  cohort=json.loads((ROOT/'campaigns/motionfix_target_semantic_v1/tasks.json').read_text())
  old=json.loads((ROOT/'campaigns/luna_extend30_v1/combined_report.json').read_text())['items']
  assert [x['key'] for x in old]==[x['key'] for x in cohort[:30]]
  tasks=cohort[30:100];assert len(tasks)==70
  manifest=[]
  for row in tasks:
   a=json.loads(read(row['A']['result']));ret=json.loads(read(a['retarget_report']))
   assert ret['key']==row['key'] and a['instruction']==row['instruction']
   read(ret['human']);row['pregmr_human']=ret['human']
   manifest.append({'key':row['key'],'instruction':row['instruction'],'source':ret['human']})
  OUT.mkdir(exist_ok=False)
  write(OUT/'inputs.json',tasks);write(OUT/'human_manifest.json',manifest)
 elif stage=='references':
  from visualmimic.luna_abc_prepare import arrays,render_reference
  tasks=json.loads((OUT/'inputs.json').read_text())
  for row in tasks:
   candidates={}
   for group in ('A','B'):
    r=json.loads(read(row[group]['result']))
    assert r['key']==row['key'] and r['instruction']==row['instruction'] and r['rollout']['outcome']=='completed'
    ar=arrays(r['rollout']['trajectory']);v=r['render']
    candidates[group]={'video':v['video'],'timestamps_s':ar['timestamps_s'][v['state_indices']].tolist(),'config':v['config'],'source_result':row[group]['result']}
    if group=='A':a=r
   assert candidates['A']['config']==candidates['B']['config']
   candidates['C']=render_reference(a,OUT/'reference_videos'/f"{row['key']['sample_id']}.mp4")
   row['candidates']=candidates
   print(row['key']['sample_id'],'reference prepared',flush=True)
  write(OUT/'reference_tasks.json',tasks)
 elif stage=='finalize':
  tasks=json.loads((OUT/'reference_tasks.json').read_text())
  rendered=json.loads((OUT/'render/manifest.json').read_text())
  verify_pre_gmr_videos(tasks,rendered,OUT/'render_preparation')
  videos={json.dumps(x['key'],sort_keys=True):x for x in rendered}
  assert len(tasks)==len(videos)==70
  for row in tasks:
   v=videos[json.dumps(row['key'],sort_keys=True)]
   assert v['instruction']==row['instruction'];read(v['video']);row['candidates']['pre_GMR']=v
  write(OUT/'tasks.json',tasks)
  for index in range(7):write(OUT/'shards'/str(index)/'tasks.json',tasks[index*10:(index+1)*10])
 elif stage=='summary':
  reports=[ROOT/'campaigns/luna_abc10_v1/report.json',ROOT/'campaigns/luna_pregmr10_v1/report.json',ROOT/'campaigns/luna_extend30_v1/report.json']+[OUT/'shards'/str(i)/'report.json' for i in range(7)]
  items=[r for p in reports for r in json.loads(p.read_text())['items']]
  tasks=json.loads((ROOT/'campaigns/motionfix_target_semantic_v1/tasks.json').read_text())[:100]
  if len(tasks)!=100:raise ValueError('Expected 100 cohort tasks')
  rows,counts=combine_judgments(tasks,items)
  transitions=dict(Counter(x['pre_GMR']['verdict']+' -> '+x['C']['verdict'] for x in rows))
  write(OUT/'combined_report.json',{'samples':100,'judgments':400,'groups':counts,'pre_GMR_to_C':transitions,'items':rows,'source_reports':[str(p) for p in reports]})
  print(json.dumps({'groups':counts,'pre_GMR_to_C':transitions}),flush=True)
 else:raise ValueError(stage)
if __name__=='__main__':main(sys.argv[1])
