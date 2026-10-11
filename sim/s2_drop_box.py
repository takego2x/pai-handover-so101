"""S2: 床に箱を落とす最小シーン。1 ステップ = 0.01 秒。"""
import genesis as gs

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.30)))
scene.build()

for i in range(101):
    if i % 20 == 0:
        print(f"t={i * 0.01:.2f}s  z={float(box.get_pos()[2]):.3f} m")
    scene.step()
