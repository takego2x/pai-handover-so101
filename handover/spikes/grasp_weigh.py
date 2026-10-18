import numpy as np, genesis as gs, torch, math, sys
gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
              rigid_options=gs.options.RigidOptions(box_box_detection=True, friction_cone=gs.friction_cone.elliptic, impratio=10.0, batch_links_info=True), show_viewer=False)
sc.add_entity(gs.morphs.Plane())
arm = sc.add_entity(gs.morphs.MJCF(file="/m/Simulation/SO101/so101_new_calib.xml", decompose_robot_error_threshold=0.0), material=gs.materials.Rigid(friction=1.0))
H = 0.0  # 床の上
cube = sc.add_entity(gs.morphs.Box(size=(0.03,)*3, pos=(0.239, 0.0, H+0.015)), material=gs.materials.Rigid(friction=1.0, rho=1000.0))
N=5; sc.build(n_envs=N)   # env0: キューブなし（遠くへ）、env1: 台の上のキューブ（27 g）、env2: 同じく 108 g
arm.set_dofs_kp(np.full(6, 998.22)); arm.set_dofs_kv(np.full(6, 2.731)); arm.set_dofs_force_range(-np.full(6,2.94), np.full(6,2.94))
cube.set_pos(torch.tensor([[0.239,0,0.015]]*N, dtype=gs.tc_float))
MASS=np.array([0.020,0.050,0.100,0.150,0.200]); cube.set_mass(torch.tensor(MASS, dtype=gs.tc_float))
grip = arm.get_link("gripper"); OFF = np.array([-0.0079, -0.000218, -0.0981])
def rot(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def quat_of(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2
    return np.array([w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)])
def Ry(a): return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
arm.set_dofs_position(np.zeros((N,6))); arm.control_dofs_position(np.zeros((N,6)))
for _ in range(30): sc.step()
R0 = rot(grip.get_quat().cpu().numpy()[0]); Rd = Ry(math.pi/2) @ R0
last=None
def go(target, n, g):
    global last
    tg=np.tile(np.asarray(target,float),(N,1)); start = last.copy() if last is not None else tg.copy()
    for k in range(n):
        s_=min(1.0,(k+1)/(0.7*n)); s_=3*s_**2-2*s_**3; cur=start+(tg-start)*s_
        if k%5==0 or k==n-1:
            q=arm.inverse_kinematics(link=grip,pos=cur-Rd@OFF,quat=np.tile(quat_of(Rd),(N,1)),init_qpos=arm.get_dofs_position(),max_solver_iters=50).cpu().numpy(); q[:,5]=g; arm.control_dofs_position(q)
        sc.step()
    last=tg.copy()
def tips():
    P=grip.get_pos().cpu().numpy(); Q=grip.get_quat().cpu().numpy(); return np.array([P[i]+rot(Q[i])@OFF for i in range(N)])
T=np.array([0.22,0,0.020])
go(T+[0,0,0.08],150,1.2); go(T,150,1.2)
print("down tip z", tips()[:,2].round(3))
v = grip.geoms[0].get_verts().cpu().numpy()
print("fixed finger geom0 verts shape", v.shape)
for e in range(N):
    ve = v[e] if v.ndim==3 else v
    print(f"  env{e} fixed finger x {ve[:,0].min():.4f}..{ve[:,0].max():.4f}  z {ve[:,2].min():.4f}..{ve[:,2].max():.4f}  tip {tips()[e].round(4)}  q {arm.get_dofs_position().cpu().numpy()[e].round(2)}")
names = {l.idx: l.name for l in arm.links}; names[cube.links[0].idx]="cube"
go(T,100,-0.17); tau_before=arm.get_dofs_control_force().cpu().numpy()
c = arm.get_contacts(with_entity=cube); la,lb,pos,fo=[x.cpu().numpy() for x in (c["link_a"],c["link_b"],c["position"],c["force_a"])]; vm=c["valid_mask"].cpu().numpy()
print("after close, env1 contacts:")
for j in range(la.shape[1]):
    if vm[1,j]: print("   ", names.get(int(la[1,j])), "<->", names.get(int(lb[1,j])), np.round(pos[1,j],4), "force", np.round(fo[1,j],2))
cf = cube.get_contacts(); print("cube contacts (all) env1 count", int(cf["valid_mask"].cpu().numpy()[1].sum()))
print("grip q", arm.get_dofs_position().cpu().numpy()[:,5].round(2))
go(T+[0,0,0.06],150,-0.17); go(T+[0,0,0.06],50,-0.17)
taus=[]
for _ in range(100):
    sc.step(); taus.append(arm.get_dofs_control_force().cpu().numpy())
tm=np.mean(taus,axis=0); ts=np.std(taus,axis=0)
print("cube z", cube.get_pos().cpu().numpy()[:,2].round(3))
for i,m in enumerate(MASS): print(f"{m*1000:5.0f} g  tau_lift {tm[i,1]:+.4f} (sd {ts[i,1]:.4f})  tau_elbow {tm[i,2]:+.4f}  tau_wrist {tm[i,3]:+.4f}")
A=np.vstack([np.ones(len(MASS)),MASS]).T; c,_,_,_=np.linalg.lstsq(A,tm[:,1],rcond=None); print("tau_lift = %.4f + %.4f * m   residual max %.4f Nm -> %.1f g" % (c[0],c[1],np.abs(A@c-tm[:,1]).max(), np.abs(A@c-tm[:,1]).max()/abs(c[1])*1000))
