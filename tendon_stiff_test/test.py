import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_string("""
<mujoco model="1-joint-2-tendon-leg">
    <option timestep="0.005" iterations="50" solver="Newton" tolerance="1e-20"/>
    
    <size njmax="200" nconmax="100" nstack="200"/>

    <default>
        <joint type="hinge" axis="0 1 0" limited="true" range="-45 45" damping="5" frictionloss=".2" />
        <geom type="capsule" size="0.05" rgba="0.6 0.6 0.6 1"/>
        <tendon stiffness="300" damping="1"/>
    </default>

    <worldbody>
          <camera name="default" pos="0 -2 1" xyaxes="1 0 0 0 0 1"/>

        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" ambient="0.1 0.1 0.1"/>
        <light pos="2 2 1" dir="-1 -1 0" diffuse="0.6 0.6 0.6"/>
        <geom name="floor" type="plane" pos="0 0 -1" size="2 2 0.1" rgba="0.222 0.3 0.2 1"/>

        <body name="base" pos="0 0 -.4">

            <geom type="box" size="0.05 0.05 0.05"  mass="1" contype="0" conaffinity="0" />

            <site name="base_flexor" pos="-0.1 0.1 0.1" size="0.01"/>
            <site name="base_extensor" pos="0.1 0.1 0.1" size="0.01"/>

            <body name="leg" pos="0 0 0">
                <joint name="hip" axis="0 1 0" range="-45 45" />
                <geom name="hip" type="cylinder" fromto="0 0 0  0 0 -0.5" size="0.02" mass="3" />

                <site name="joint_left" pos="-0.12 0 0" size="0.01"/>
                <site name="joint_right" pos="0.12 0 0" size="0.01"/>
                <site name="leg_flexor" pos="-0.1 0 -0.15" size="0.01"/>
                <site name="leg_extensor" pos="0.1 0 -0.15" size="0.01"/>
            </body>
        </body>
        </worldbody>

      <tendon>
            <spatial name="tendon_flexor" >
                <site site="base_flexor"/>
                <site site="joint_left"/>
                <site site="leg_flexor"/>
            </spatial>
            <spatial name="tendon_extensor" >
                <site site="base_extensor"/>
                <site site="joint_right"/>

                <site site="leg_extensor"/>
            </spatial>
        </tendon>



    <actuator>
        <motor name="flexor" tendon="tendon_flexor" gear="-50" ctrlrange="0 1" forcerange="0 50"/>
        <motor name="extensor" tendon="tendon_extensor" gear="-50" ctrlrange="0 1" forcerange="0 50"/>
    </actuator>
    
</mujoco>""")



data = mujoco.MjData(model)

data = mujoco.MjData(model)

# Actuator and joint IDs
flexor_id = model.actuator("flexor").id
extensor_id = model.actuator("extensor").id
hip_qpos_id = model.joint("hip").qposadr[0]

# Data storage
angles_deg = []
forces_flexor = []
forces_extensor = []
time_log = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = data.time
    while viewer.is_running():
        t = data.time - start_time

        if t <= 5.0:
            ctrl_flexor = 0.4 * np.sin(2 * np.pi * 0.5 * t)
            ctrl_extensor = 0.4 * np.sin(2 * np.pi * 0.5 * t + np.pi)

            data.ctrl[flexor_id] = np.clip(ctrl_flexor, 0, 1)
            data.ctrl[extensor_id] = np.clip(ctrl_extensor, 0, 1)

        mujoco.mj_step(model, data)

        time_log.append(t)
        angles_deg.append(np.rad2deg(data.qpos[hip_qpos_id]))
        forces_flexor.append(data.actuator_force[flexor_id])
        forces_extensor.append(data.actuator_force[extensor_id])

        viewer.sync()

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Subplot 1: Hip angle in degrees
ax1.plot(time_log, angles_deg, label="Hip Angle (°)")
ax1.set_ylabel("Angle (°)")
ax1.set_title("Joint Angle Over Time")
ax1.legend()
ax1.grid(True)

forces_flexor  = [i*50 for i in forces_flexor ]
forces_extensor  = [i*50 for i in forces_extensor ]

# Subplot 2: Tendon forces
ax2.plot(time_log, forces_flexor, label="Flexor Force (N)")
ax2.plot(time_log, forces_extensor, label="Extensor Force (N)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Force (N)")
ax2.set_title("Tendon Forces Over Time")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
