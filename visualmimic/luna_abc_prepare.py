"""Prepare matched A/B execution and C kinematic reference clips on ICE."""
import io, json, hashlib
from pathlib import Path
import numpy as np
from visualmimic.luna_target_pilot import ROOT, read, write
OUT=ROOT/'campaigns/luna_abc10_v1'

def arrays(ref):
 with np.load(io.BytesIO(read(ref)),allow_pickle=False) as z:return dict(z)

def render_reference(result, output):
 import mujoco
 import imageio.v2 as imageio
 from supermimic.data.motion_convert import g1_xml_path
 cfg=result['render']['config'];xml=Path(g1_xml_path())
 assert hashlib.sha256(xml.read_bytes()).hexdigest()==result['robot']['model_asset']['sha256']
 ref=result['reference'];a=arrays(ref['motion_npz'])
 assert a['joint_names'].tolist()==result['robot']['joint_names']
 assert a['body_names'].tolist()==result['robot']['body_names']
 assert a['body_names'][0]=='pelvis'
 spec=mujoco.MjSpec.from_file(str(xml))
 spec.worldbody.add_geom(name='video_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],rgba=[.22,.25,.30,1])
 spec.worldbody.add_light(pos=[0,-3,5],dir=[0,0,-1])
 model=spec.compile();model.vis.global_.offwidth=cfg['width'];model.vis.global_.offheight=cfg['height']
 assert [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,model.njnt)]==a['joint_names'].tolist()
 data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=cfg['height'],width=cfg['width'])
 cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
 cam.azimuth=cfg['azimuth'];cam.elevation=cfg['elevation'];cam.distance=cfg['distance'];cam.lookat[:]=[*a['body_pos_w'][0,0,:2],.65]
 option=mujoco.MjvOption();option.geomgroup[3:]=False
 fps=float(a['fps'].reshape(-1)[0]);stride=round(fps/cfg['fps']);assert np.isclose(stride*cfg['fps'],fps)
 ids=list(range(0,len(a['joint_pos']),stride))
 if ids[-1]!=len(a['joint_pos'])-1:ids.append(len(a['joint_pos'])-1)
 assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
 writer=imageio.get_writer(str(output),fps=cfg['fps'],codec='libx264',quality=8)
 max_fk_error=0.
 try:
  for i in ids:
   data.qpos[:]=np.concatenate([a['body_pos_w'][i,0],a['body_quat_w'][i,0],a['joint_pos'][i]])
   data.qvel[:]=0;mujoco.mj_forward(model,data)
   err=float(np.max(np.abs(data.xpos[1:]-a['body_pos_w'][i])));max_fk_error=max(max_fk_error,err)
   assert err<1e-4,(i,err)
   if cfg['camera']=='root-follow':cam.lookat[:2]=data.qpos[:2]
   renderer.update_scene(data,camera=cam,scene_option=option);writer.append_data(renderer.render().copy())
 finally:writer.close();renderer.close()
 return {'video':{'uri':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'media_type':'video/mp4','schema_version':'1'},'timestamps_s':(np.array(ids)/fps).tolist(),'config':cfg,'source_reference':ref,'physics_execution':False,'max_fk_position_error_m':max_fk_error}

def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'tasks.json').exists():raise RuntimeError('Preparation already exists')
 tasks=json.loads((ROOT/'campaigns/luna_target10_v1/tasks.json').read_text());assert len(tasks)==10
 for row in tasks:
  candidates={}
  for group in ('A','B'):
   r=json.loads(read(row[group]['result']));assert r['key']==row['key'] and r['instruction']==row['instruction']
   assert r['rollout']['outcome']=='completed'
   ar=arrays(r['rollout']['trajectory']);v=r['render']
   candidates[group]={'video':v['video'],'timestamps_s':ar['timestamps_s'][v['state_indices']].tolist(),'config':v['config'],'source_result':row[group]['result'],'physics_execution':True}
   if group=='A':a=r
  assert candidates['A']['config']==candidates['B']['config']
  candidates['C']=render_reference(a,OUT/'reference_videos'/f"{row['key']['sample_id']}.mp4")
  row['candidates']=candidates
  print(row['key']['sample_id'],'prepared',flush=True)
 write(OUT/'tasks.json',tasks)
if __name__=='__main__':main()
