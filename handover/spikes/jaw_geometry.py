import numpy as np, genesis as gs, math, sys
gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=10), show_viewer=False)
sc.add_entity(gs.morphs.Plane())
arm = sc.add_entity(gs.morphs.MJCF(file="/m/Simulation/SO101/so101_new_calib.xml", decompose_robot_error_threshold=0.0))
sc.build()
arm.set_dofs_kp(np.full(6, 998.22)); arm.set_dofs_kv(np.full(6, 2.731)); arm.set_dofs_force_range(-np.full(6,2.94), np.full(6,2.94))
grip = arm.get_link("gripper"); jaw = arm.get_link("moving_jaw_so101_v1")
OFF = np.array([-0.0079, -0.000218, -0.0981])
def rot(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def quat_of(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2
    return np.array([w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)])
def Ry(a): return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
arm.set_dofs_position(np.zeros(6)); arm.control_dofs_position(np.zeros(6))
for _ in range(30): sc.step()
R0 = rot(grip.get_quat().cpu().numpy()); Rd = Ry(math.pi/2) @ R0
T = np.array([0.22, 0.0, 0.08])
for g in [1.2, -0.17]:
    q = arm.inverse_kinematics(link=grip, pos=T - Rd @ OFF, quat=quat_of(Rd), init_qpos=arm.get_dofs_position()).cpu().numpy(); q[5]=g
    arm.control_dofs_position(q)
    for _ in range(200): sc.step()
    P=grip.get_pos().cpu().numpy(); Q=grip.get_quat().cpu().numpy(); tip=P+rot(Q)@OFF
    print(f"== gripper q {arm.get_dofs_position().cpu().numpy()[5]:+.2f}  target {T}  tip(site) {tip.round(4)}")
    for name, link in [("fixed(gripper link)", grip), ("moving jaw", jaw)]:
        for gi, geom in enumerate(link.geoms):
            v = geom.get_verts().cpu().numpy().reshape(-1,3)
            print(f"  {name} geom{gi}: n={len(v)} min {v.min(0).round(4)} max {v.max(0).round(4)}")
