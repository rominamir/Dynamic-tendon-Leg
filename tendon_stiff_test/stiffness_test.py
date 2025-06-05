import mujoco
import numpy as np
import matplotlib.pyplot as plt

model_xml = """
<mujoco model="tendon_stretch_test">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <body name="anchor" pos="0 0 0">
      <geom type="sphere" size="0.01" rgba="1 0 0 1"/>
      <site name="anchor_site" pos="0 0 0" size="0.005" />
    </body>

    <body name="weight" pos="0 0 -0.2">
      <joint name="free" type="free" damping="2"/>

      <geom type="sphere" size="0.03" mass="1"/>
      <site name="mass_site" pos="0 0 0" size="0.005" />
    </body>
  </worldbody>

  <tendon>
    <spatial name="spring_tendon" stiffness="490.5" springlength="0.2">
      <site site="anchor_site"/>
      <site site="mass_site"/>
    </spatial>
  </tendon>

  <actuator/>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "spring_tendon")
rest_length = 0.2
stiffness = 490.5

sim_time = 2.0
dt = model.opt.timestep
n_steps = int(sim_time / dt)

lengths = []
times = []

for i in range(n_steps):
    mujoco.mj_step(model, data)
    length = data.ten_length[tendon_id]
    lengths.append(length)
    times.append(i * dt)

plt.figure(figsize=(8, 5))
plt.plot(times, lengths, label="Tendon Length")
plt.axhline(y=0.2, color='gray', linestyle='--', label="Rest Length (0.2m)")
plt.axhline(y=0.22, color='red', linestyle='--', label="Target Length (0.22m)")
plt.xlabel("Time (s)")
plt.ylabel("Tendon Length (m)")
plt.title("Tendon Length Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()




forces = []
lengths = []
times = []

for i in range(n_steps):
    mujoco.mj_step(model, data)

    # Get current tendon length
    length = data.ten_length[tendon_id]
    lengths.append(length)

    # Compute tendon force manually (spring only acts when stretched)
    force = stiffness * (length - rest_length) if length > rest_length else 0.0
    forces.append(force)

    # Store time
    times.append(i * dt)


plt.figure(figsize=(8, 5))
plt.plot(times, forces, label="Tendon Force (N)")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Tendon Force Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
