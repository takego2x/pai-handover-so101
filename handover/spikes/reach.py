import numpy as np, genesis as gs, math, itertools
gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10), show_viewer=False)
sc.add_entity(gs.morphs.Plane())
arm = sc.add_entity(gs.morphs.MJCF(file="/m/Simulation/SO101/so101_new_calib.xml", decompose_robot_error_threshold=0.0))
T = list(itertools.product([0.16,0.20,0.24,0.28,0.32],[0.02,0.04,0.08]))
N=len(T); sc.build(n_envs=N)
arm.set_dofs_kp(np.full(6, 998.22)); arm.set_dofs_kv(np.full(6, 2.731)); arm.set_dofs_force_range(-np.full(6,2.94), np.full(6,2.94))
grip = arm.get_link("gripper")
OFF = np.array([-0.0079, -0.000218, -0.0981])
def rot(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def quat_of(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2
    return np.array([w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)])
def Ry(a): return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
arm.set_dofs_position(np.zeros((N,6))); arm.control_dofs_position(np.zeros((N,6)))
for _ in range(30): sc.step()
R0 = rot(grip.get_quat().cpu().numpy()[0]); Rd = Ry(math.pi/2) @ R0
tg = np.array([[x,0,z] for x,z in T])
for it in range(3):  # 少しずつ近づける（IK を何度か）
    q = arm.inverse_kinematics(link=grip, pos=tg - Rd @ OFF, quat=np.tile(quat_of(Rd),(N,1)), init_qpos=arm.get_dofs_position(), max_solver_iters=200).cpu().numpy()
    q[:,5]=1.2; arm.control_dofs_position(q)
    for _ in range(100): sc.step()
P=grip.get_pos().cpu().numpy(); Q=grip.get_quat().cpu().numpy()
lo,hi=[x.cpu().numpy() for x in arm.get_dofs_limit()]
qa=arm.get_dofs_position().cpu().numpy()
for i,(x,z) in enumerate(T):
    tip=P[i]+rot(Q[i])@OFF; down=(rot(Q[i])@np.array([0,0,-1]))[2]
    lim=[n for n,(v,l,h) in enumerate(zip(qa[i,:5],lo[:5],hi[:5])) if v<l+0.03 or v>h-0.03]
    print(f"target x {x:.2f} z {z:.2f}  tip {tip.round(3)}  err {np.linalg.norm(tip-tg[i])*1000:5.1f}mm  tilt-from-down {math.degrees(math.acos(max(-1,min(1,-down)))):5.1f}deg  at-limit {lim}")
