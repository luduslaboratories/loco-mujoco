from typing import List, Union, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class MJHumanoid(BaseRobotHumanoid):

    """
    Description
    ------------
    Humanoid model based on the standard MuJoCo humanoid.
    """

    mjx_enabled = False

    def __init__(self,
                 spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None,
                 **kwargs) -> None:
        """
        Constructor.

        Args:
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[Observation], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
        """


        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)


        super().__init__(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec, **kwargs)

    def _get_spec_modifications(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Specifies which joints, actuators, and equality constraints should be removed from the Mujoco specification.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing lists of joints to remove, actuators to remove,
            and equality constraints to remove.
        """

        joints_to_remove = []
        actuators_to_remove = []
        equ_constr_to_remove = []


        return joints_to_remove, actuators_to_remove, equ_constr_to_remove


    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: List of observations.
        """
        observation_spec = [# ------------- JOINT POS -------------
                            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
                            # Abdomen/Torso joints
                            ObservationType.JointPos("q_abdomen_z", xml_name="abdomen_z"),
                            ObservationType.JointPos("q_abdomen_y", xml_name="abdomen_y"),
                            ObservationType.JointPos("q_abdomen_x", xml_name="abdomen_x"),
                            # Left leg
                            ObservationType.JointPos("q_left_hip_x", xml_name="hip_x_left"),
                            ObservationType.JointPos("q_left_hip_z", xml_name="hip_z_left"),
                            ObservationType.JointPos("q_left_hip_y", xml_name="hip_y_left"),
                            ObservationType.JointPos("q_left_knee", xml_name="knee_left"),
                            ObservationType.JointPos("q_left_ankle_y", xml_name="ankle_y_left"),
                            ObservationType.JointPos("q_left_ankle_x", xml_name="ankle_x_left"),
                            # Right leg
                            ObservationType.JointPos("q_right_hip_x", xml_name="hip_x_right"),
                            ObservationType.JointPos("q_right_hip_z", xml_name="hip_z_right"),
                            ObservationType.JointPos("q_right_hip_y", xml_name="hip_y_right"),
                            ObservationType.JointPos("q_right_knee", xml_name="knee_right"),
                            ObservationType.JointPos("q_right_ankle_y", xml_name="ankle_y_right"),
                            ObservationType.JointPos("q_right_ankle_x", xml_name="ankle_x_right"),
                            # Left arm
                            ObservationType.JointPos("q_left_shoulder1", xml_name="shoulder1_left"),
                            ObservationType.JointPos("q_left_shoulder2", xml_name="shoulder2_left"),
                            ObservationType.JointPos("q_left_elbow1", xml_name="elbow1_left"),
                            ObservationType.JointPos("q_left_elbow2", xml_name="elbow2_left"),
                            # Right arm
                            ObservationType.JointPos("q_right_shoulder1", xml_name="shoulder1_right"),
                            ObservationType.JointPos("q_right_shoulder2", xml_name="shoulder2_right"),
                            ObservationType.JointPos("q_right_elbow1", xml_name="elbow1_right"),
                            ObservationType.JointPos("q_right_elbow2", xml_name="elbow2_right"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            # Abdomen/Torso joints
                            ObservationType.JointVel("dq_abdomen_z", xml_name="abdomen_z"),
                            ObservationType.JointVel("dq_abdomen_y", xml_name="abdomen_y"),
                            ObservationType.JointVel("dq_abdomen_x", xml_name="abdomen_x"),
                            # Left leg
                            ObservationType.JointVel("dq_left_hip_x", xml_name="hip_x_left"),
                            ObservationType.JointVel("dq_left_hip_z", xml_name="hip_z_left"),
                            ObservationType.JointVel("dq_left_hip_y", xml_name="hip_y_left"),
                            ObservationType.JointVel("dq_left_knee", xml_name="knee_left"),
                            ObservationType.JointVel("dq_left_ankle_y", xml_name="ankle_y_left"),
                            ObservationType.JointVel("dq_left_ankle_x", xml_name="ankle_x_left"),
                            # Right leg
                            ObservationType.JointVel("dq_right_hip_x", xml_name="hip_x_right"),
                            ObservationType.JointVel("dq_right_hip_z", xml_name="hip_z_right"),
                            ObservationType.JointVel("dq_right_hip_y", xml_name="hip_y_right"),
                            ObservationType.JointVel("dq_right_knee", xml_name="knee_right"),
                            ObservationType.JointVel("dq_right_ankle_y", xml_name="ankle_y_right"),
                            ObservationType.JointVel("dq_right_ankle_x", xml_name="ankle_x_right"),
                            # Left arm
                            ObservationType.JointVel("dq_left_shoulder1", xml_name="shoulder1_left"),
                            ObservationType.JointVel("dq_left_shoulder2", xml_name="shoulder2_left"),
                            ObservationType.JointVel("dq_left_elbow1", xml_name="elbow1_left"),
                            ObservationType.JointVel("dq_left_elbow2", xml_name="elbow2_left"),
                            # Right arm
                            ObservationType.JointVel("dq_right_shoulder1", xml_name="shoulder1_right"),
                            ObservationType.JointVel("dq_right_shoulder2", xml_name="shoulder2_right"),
                            ObservationType.JointVel("dq_right_elbow1", xml_name="elbow1_right"),
                            ObservationType.JointVel("dq_right_elbow2", xml_name="elbow2_right")
                            ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: List of action names.
        """
        action_spec = [
            # Torso
            "abdomen_z", "abdomen_y", "abdomen_x",
            # Left leg
            "hip_x_left", "hip_z_left", "hip_y_left", "knee_left", "ankle_y_left", "ankle_x_left",
            # Right leg
            "hip_x_right", "hip_z_right", "hip_y_right", "knee_right", "ankle_y_right", "ankle_x_right",
            # Left arm
            "shoulder1_left", "shoulder2_left", "elbow1_left", "elbow2_left",
            # Right arm
            "shoulder1_right", "shoulder2_right", "elbow1_right", "elbow2_right",
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the MJHumanoid environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "MJHumanoid" / "humanoid.xml").as_posix()

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body specified in the XML file.
        """
        return "torso"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root specified in the XML file.
        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        """
        return (0.4, 1.5)


    # @info_property
    # def sites_for_mimic(self) -> List[str]:
    #     """
    #     Returns the default sites that are used for mimic.
    #     """
    #     return ["upper_body_mimic", "head_mimic", "pelvis_mimic",
    #             "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic",
    #             "left_hip_mimic", "left_knee_mimic", "left_foot_mimic",
    #             "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
    #             "right_hip_mimic", "right_knee_mimic", "right_foot_mimic"]