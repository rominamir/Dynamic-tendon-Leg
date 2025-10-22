import mujoco
import numpy as np

# Paste your full MuJoCo XML string here
model_xml = """
<mujoco model="1-joint-2-tendon-leg">
    <option timestep="0.005" iterations="50" solver="Newton" tolerance="1e-10"/>
    <size njmax="20" nconmax="10" nstack="200"/>

    <default>
        <joint type="hinge" axis="0 1 0" limited="true" range="-45 45" damping="0.5" frictionloss="1" />
        <geom type="capsule" size="0.05" rgba="0.6 0.6 0.6 1"/>
        <tendon stiffness="3000" damping="10"/>
        <muscle ctrllimited="true" ctrlrange="0 1" force="1000" gear="1"/>
    </default>

    <worldbody>
        <geom name="floor" type="plane" pos="0 0 -1" size="2 2 0.1" rgba="0.2 0.3 0.4 1"/>

        <!-- Base body -->
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1" rgba="0.4 0.4 0.4 1"/>

            <!-- Single leg segment with 1 joint (hip) -->
            <body name="leg" pos="0 0 0">
                <joint name="hip"/>
                <geom fromto="0 0 0  0 0 -0.4"/>

                <!-- Tendon attachment sites on the leg -->
                <site name="leg_flexor" pos="-0.05 0 -0.2" size="0.01"/>
                <site name="leg_extensor" pos="0.05 0 -0.2" size="0.01"/>
            </body>

            <!-- Tendon attachment sites on the base -->
            <site name="base_flexor" pos="-0.05 0 0.1" size="0.01"/>
            <site name="base_extensor" pos="0.05 0 0.1" size="0.01"/>
        </body>
    </worldbody>

    <tendon>
        <spatial name="tendon_flexor">
            <site site="base_flexor"/>
            <site site="leg_flexor"/>
        </spatial>
        <spatial name="tendon_extensor">
            <site site="base_extensor"/>
            <site site="leg_extensor"/>
        </spatial>
    </tendon>

    <actuator>
        <muscle name="flexor" tendon="tendon_flexor"/>
        <muscle name="extensor" tendon="tendon_extensor"/>
    </actuator>
</mujoco>
"""

# Load model
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

 
# Step the simulation once to compute tendon lengths
data.ctrl[:] = [1.0, 1.0]  # Full activation of all 3 muscles

mujoco.mj_step(model, data)
# Print tendon lengths
print("Tendon Lengths:")
for i in range(model.ntendon):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
    length = data.ten_length[i]
    print(f"  {name}: {length:.4f} m")

# Print actuator (muscle) forces
print("\nActuator (Muscle) Forces:")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    force = data.actuator_force[i]
    print(f"  {name}: {force:.4f} N")

# Create buffer for tendon forces
tendon_forces = np.zeros(model.ntendon)

# Compute tendon forces

# Print results
print("\nTendon Forces:")
for i in range(model.ntendon):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
    print(f"  {name}: {tendon_forces[i]:.4f} N")