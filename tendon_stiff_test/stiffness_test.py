import mujoco
import numpy as np
import matplotlib.pyplot as plt

model_xml = """
<mujoco model="tendon_stretch_test">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <!-- Fixed anchor -->
    <body name="anchor" pos="0 0 0">
      <geom type="sphere" size="0.01" rgba="1 0 0 1"/>
      <site name="anchor_site" pos="0 0 0" size="0.005" />
    </body>

    <!-- 10 kg mass hanging below -->
    <body name="weight" pos="0 0 -1">
      <joint name="free" type="free"/>
      <geom type="sphere" size="0.03" mass="0.6"/>
      <site name="mass_site" pos="0 0 0" size="0.005" />
    </body>
  </worldbody>

  <tendon>
    <spatial name="spring_tendon" stiffness="30000" >
      <site site="anchor_site"/>
      <site site="mass_site"/>
    </spatial>
  </tendon>

  <-- <actuator>
     <muscle name="muscle_force" tendon="spring_tendon" ctrlrange="0 1" force="5000" lengthrange="0.1 0.3"/>
    </actuator> --!>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "spring_tendon")
rest_length = 1
stiffness = 30000

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
plt.axhline(y=2, color='gray', linestyle='--', label="Rest Length (0.2m)")
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
    #force = stiffness * (length - rest_length) if length > rest_length else 0.0
    force = model.tendon_stiffness[tendon_id] * max(0.0, data.ten_length[tendon_id] - model.tendon_length0[tendon_id])

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


import mujoco
import numpy as np
import matplotlib.pyplot as plt


model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "spring_tendon")
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle_force")

# Simulation config
sim_time = 10
dt = model.opt.timestep
n_steps = int(sim_time / dt)

# Full activation of muscle
data.ctrl[actuator_id] = 1.0



# Logging
manual_forces = []
actuator_forces = []
lengths = []
times = []

for i in range(n_steps):
    mujoco.mj_step(model, data)

    length = data.ten_length[tendon_id]
    lengths.append(length)


    #print(f"Tendon length at t={i*dt:.3f}: {length:.4f}")
    # Manual spring force (passive)
    rest_length = 1

    spring_force = model.tendon_stiffness[tendon_id] *( length - rest_length)


    manual_forces.append(spring_force)

    # Muscle actuator force
    actuator_force = data.actuator_force[actuator_id]
    actuator_forces.append(actuator_force)

    times.append(i * dt)
"otting"
plt.figure(figsize=(10, 6))
plt.plot(times, manual_forces, label="Manual Tendon Spring Force (N)")
plt.plot(times, actuator_forces, label="Actuator (Muscle) Force (N)", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Tendon Spring vs Muscle Actuator Force")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
