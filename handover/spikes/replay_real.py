import numpy as np, pyarrow.parquet as pq, genesis as gs, math
gs.init(backend=gs.cpu, logging_level="warning")
sc = gs.Scene(sim_options=gs.options.SimOptions(dt=1/30), show_viewer=False)
sc.add_entity(gs.morphs.Plane())
arm = sc.add_entity(gs.morphs.MJCF(file="/m/Simulation/SO101/so101_new_calib.xml"))
cam = sc.add_camera(res=(640, 480), pos=(0.45, -0.35, 0.35), lookat=(0.12, 0.0, 0.08), fov=50, GUI=False)
sc.build()
names = [j.name for j in arm.joints if j.dof_idx_local is not None]
print("joints:", names, "n_dofs", arm.n_dofs)
lo, hi = [x.cpu().numpy() for x in arm.get_dofs_limit()]
print("limits deg:", np.degrees(lo).round(0), np.degrees(hi).round(0))
t = pq.read_table("/d/data/chunk-000/file-000.parquet")
s = np.array(t.column("observation.state").to_pylist()); ep = np.array(t.column("episode_index").to_pylist())
e = s[ep == 0]
q = np.radians(e[:, :5])  # 角度は度とみなす（要確認）
g = lo[5] + (hi[5] - lo[5]) * e[:, 5] / 100.0  # 指は 0〜100 → 関節の範囲（要確認）
qq = np.column_stack([q, g])
cam.start_recording(save_to_filename="/out/replay_ep0.mp4", fps=30)
for k in range(len(qq)):
    arm.set_dofs_position(qq[k])
    sc.step()
cam.stop_recording()
print("frames", len(qq))
