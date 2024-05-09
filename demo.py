import matplotlib.pyplot as plt
import numpy as np
from abr_control.utils.transformations import quaternion_from_euler

from mausspaun.arm import MouseArm

mousearm = MouseArm(
    use_muscles=False,
    use_sim_state=False,
    xml_kwargs={
        "articulated_hand": False,
        "add_rig": True,
        "add_rig_markers": False,
        "add_joystick": False,
        "add_joint_targets": True
    },
    visualize=False,
)
# q = np.array([q0, q1w, q1x, q1y, q1z, q2w, q2x, q2y, q2z, q3])
# q = np.array([scapula slide, shoulder quaternion, elbow quaternion, wrist hinge])

# to match the notation they use, we could define the two ball joints using euler
# angles instead. just have to transform the euler angles to a quaternion representation
# before sending it into the Tx function
# q = np.array([theta_0, phi_1, theta_1, psi_1, phi_2, theta_2, psi_2, theta_3])
q = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])


def euler_rep_to_quat_rep(q_euler):
    q_quat = np.zeros(10)
    q_quat[0] = q_euler[0]
    q_quat[1:5] = quaternion_from_euler(*q_euler[1:4], axes="rxyz")
    q_quat[5:9] = quaternion_from_euler(*q_euler[4:7], axes="rxyz")
    q_quat[9] = q_euler[8]
    return q_quat


q = euler_rep_to_quat_rep(q)

p_shoulder = mousearm.config.Tx("shoulder", object_type="body", q=q)
p_elbow = mousearm.config.Tx("elbow", object_type="body", q=q)
p_wrist = mousearm.config.Tx("wrist", object_type="body", q=q)
p_wrist_top = mousearm.config.Tx("wrist_top", object_type="body", q=q)
p_wrist_bottom = mousearm.config.Tx("wrist_bottom", object_type="body", q=q)
p_backofhand = mousearm.config.Tx("backofhand", object_type="body", q=q)

p_finger0_0 = mousearm.config.Tx("finger0_0_dlc", object_type="body", q=q)
p_finger0_1 = mousearm.config.Tx("finger0_1_dlc", object_type="body", q=q)
p_finger0_2 = mousearm.config.Tx("finger0_2_dlc", object_type="body", q=q)
p_finger0_3 = mousearm.config.Tx("finger0_3_dlc", object_type="body", q=q)

p_finger1_0 = mousearm.config.Tx("finger1_0_dlc", object_type="body", q=q)
p_finger1_1 = mousearm.config.Tx("finger1_1_dlc", object_type="body", q=q)
p_finger1_2 = mousearm.config.Tx("finger1_2_dlc", object_type="body", q=q)
p_finger1_3 = mousearm.config.Tx("finger1_3_dlc", object_type="body", q=q)

p_finger2_0 = mousearm.config.Tx("finger2_0_dlc", object_type="body", q=q)
p_finger2_1 = mousearm.config.Tx("finger2_1_dlc", object_type="body", q=q)
p_finger2_2 = mousearm.config.Tx("finger2_2_dlc", object_type="body", q=q)
p_finger2_3 = mousearm.config.Tx("finger2_3_dlc", object_type="body", q=q)

p_finger3_0 = mousearm.config.Tx("finger3_0_dlc", object_type="body", q=q)
p_finger3_1 = mousearm.config.Tx("finger3_1_dlc", object_type="body", q=q)
p_finger3_2 = mousearm.config.Tx("finger3_2_dlc", object_type="body", q=q)
p_finger3_3 = mousearm.config.Tx("finger3_3_dlc", object_type="body", q=q)
