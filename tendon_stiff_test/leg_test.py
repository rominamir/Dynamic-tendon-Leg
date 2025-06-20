import mujoco
import numpy as np

# Paste your full MuJoCo XML string here
model_xml = """
<mujoco model="2-link 6-muscle arm">
    <!--  Copyright © 2018, Roboti LLC

          This file is licensed under the MuJoCo Resource License (the "License").
          You may not use this file except in compliance with the License.
          You may obtain a copy of the License at

            https://www.roboti.us/resourcelicense.txt
    -->

    <option timestep="0.005" iterations="50" solver="Newton" tolerance="1e-10"/>

    <size njmax="50" nconmax="10" nstack="200"/>

    <visual>
        <rgba haze=".3 .3 .3 1"/>
    </visual>

    <default>
        <joint type="hinge" pos="0 0 0" axis="0 1 0" limited="true" damping="1" frictionloss="3" />
        <tendon stiffness = "1000"/>
        <muscle ctrllimited="true" ctrlrange="0 1" force="6000"/>
    </default>


    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.6 0.6 0.6" rgb2="0 0 0" width="512" height="512"/> 

        <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .25 .25" rgb2=".3 .3 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>  

        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    </asset>

    <worldbody>    
        <geom name="floor" pos="0 0 -0.5" size="0 0 1" type="plane" material="matplane"/>

        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 1 -5"/>
        <body name="Chassis" pos="0 0 -.05">
            <camera name="Chassis_camera" pos="0 -6 3"  zaxis="0 -2 1"/>
            <geom name="Chassis_frame" type="box" pos="0 0 1.7" zaxis="0 1 0" size=".50 .30 .25" rgba=".25 .25 .25 0.25" mass=".1"/>
  
           <joint armature="0"  axis="1 0 0" damping="0" frictionloss="30" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide" ref = "0"/>
           <!--   <joint armature="0" axis="0 0 1" damping="0" frictionloss="3" limited="false" name="rootz" pos="0 0 0" stiffness="0" springdamper=".01 10" type="slide"/>--> <!-- to enable z axis slide with a spring damper-->

            <site name="S_M0" pos="0.40 0 1.8" size="0.02"/>
            <site name="S_M2" pos="0.40 0 1.5" size="0.02"/>
            <site name="S_M1" pos="-0.40 0 1.5" size="0.02"/>

            <body pos="0 0 .5">
                <joint name="Hip" range="-60 60" ref="0"/>
                <geom name="Hip" type="cylinder" pos="0 0 0" zaxis="0 1 0" size=".1 .05" rgba="1 1 1 .5" mass="4"/>
                
                <geom name="L0" type="capsule" size="0.045" fromto="0 0 0  0 0 -.50" rgba=".7 .7 .7 1"/>
                
                <site name="S_J0_0" pos="-0.15 0 0" size="0.02" rgba="0.7 0.8 0.95 .1" group="1"/>
                <site name="S_J0_1" pos="0.15 0 0" size="0.02" rgba="0.7 0.8 0.95 .1" group="1"/>

                <site name="S_L0_1" pos="0.05 0 -.1" size="0.02"/>
                <site name="S_L0_0" pos="-0.05 0 -.1" size="0.02"/>

                <site name="S_L0_2" pos="-0.1 0 -.25" size="0.02"/>
                <site name="S_L0_3" pos=" 0.00 0 -.25" size="0.02"/>
                <site name="S_L_04" pos="+0.1 0 -.25" size="0.02"/>

                <body pos="0 0 -.5">
                    <joint name="Knee" range="-90 0" ref="-45"/>
                    <geom name="Knee" type="cylinder" pos="0 0 0" zaxis="0 1 0" size=".1 .05" rgba="1 1 1 .5" mass="4"/>
                    
                    <geom name="L1" type="capsule" size="0.045" fromto="0 0 0 0.34 0 -0.34" rgba=".7 .7 .7 1"/>

                    <site name="S_J1_0" pos="-0.15 0 0" size="0.02" rgba="0.7 0.8 0.95 .1" group="1"/>
                    <site name="S_J1_1" pos="0.15 0 0" size="0.02" rgba="0.7 0.8 0.95 .1" group="1"/>
                    
                    <site name="S_L1_0" pos="-0.05 0 -.1" size="0.02"/>
                    <site name="S_L1_1" pos="0.05 0 -.1" size="0.02"/>

                </body>
            </body>
        </body>
    </worldbody>

    <tendon>
        <spatial name="T_M0" width="0.01" rgba="0.55 0.78 0.55 1">
            <site site="S_M0"/>
            <geom geom="Hip" sidesite="S_J0_0"/>
            <site site="S_L0_3"/>
            <geom geom="Knee" sidesite="S_J1_1"/>
            <site site="S_L1_1"/>
        </spatial>
        
        <spatial name="T_M1" width="0.01" rgba="0.95 0.50 0.47 1">
            <site site="S_M1"/>
            <geom geom="Hip" sidesite="S_J0_1"/>
            <site site="S_L0_1"/>
        </spatial>

        <spatial name="T_M2" width="0.01" rgba="0.45 0.49 0.83 1">
            <site site="S_M2"/>
            <geom geom="Hip" sidesite="S_J0_0"/>
            <site site="S_L0_2"/>
            <geom geom="Knee" sidesite="S_J1_0"/>
            <site site="S_L1_0"/>
        </spatial>

    </tendon>   

    <actuator>
        <muscle name="T_M0" tendon="T_M0"/>
        <muscle name="T_M1" tendon="T_M1"/>
        <muscle name="T_M2" tendon="T_M2"/>
    </actuator>
    <sensor type="tendonforce" tendon="T_M0" name="t0_force"/>
<sensor type="tendonforce" tendon="T_M1" name="t1_force"/>
<sensor type="tendonforce" tendon="T_M2" name="t2_force"/>

</mujoco>
"""

# Load model
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

 
# Step the simulation once to compute tendon lengths
data.ctrl[:] = [1.0, 1.0, 1.0]  # Full activation of all 3 muscles

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
mujoco.mj_tendonforce(model, data, tendon_forces)

# Print results
print("\nTendon Forces:")
for i in range(model.ntendon):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
    print(f"  {name}: {tendon_forces[i]:.4f} N")