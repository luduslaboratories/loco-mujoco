from typing import List, Union, Tuple
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class SMPLBoxing(BaseRobotHumanoid):

    """
    Description
    ------------



    """

    mjx_enabled = True

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
        # super().__init__(timestep=0.002, n_substeps=5, **kwargs)

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
                            ObservationType.JointPos("q_L_Hip_x", xml_name="L_Hip_x"),
                            ObservationType.JointPos("q_L_Hip_y", xml_name="L_Hip_y"),
                            ObservationType.JointPos("q_L_Hip_z", xml_name="L_Hip_z"),
                            ObservationType.JointPos("q_L_Knee_x", xml_name="L_Knee_x"),
                            ObservationType.JointPos("q_L_Knee_y", xml_name="L_Knee_y"),
                            ObservationType.JointPos("q_L_Knee_z", xml_name="L_Knee_z"),
                            ObservationType.JointPos("q_L_Ankle_x", xml_name="L_Ankle_x"),
                            ObservationType.JointPos("q_L_Ankle_y", xml_name="L_Ankle_y"),
                            ObservationType.JointPos("q_L_Ankle_z", xml_name="L_Ankle_z"),
                            ObservationType.JointPos("q_L_Toe_x", xml_name="L_Toe_x"),
                            ObservationType.JointPos("q_L_Toe_y", xml_name="L_Toe_y"),
                            ObservationType.JointPos("q_L_Toe_z", xml_name="L_Toe_z"),
                            ObservationType.JointPos("q_R_Hip_x", xml_name="R_Hip_x"),
                            ObservationType.JointPos("q_R_Hip_y", xml_name="R_Hip_y"),
                            ObservationType.JointPos("q_R_Hip_z", xml_name="R_Hip_z"),
                            ObservationType.JointPos("q_R_Knee_x", xml_name="R_Knee_x"),
                            ObservationType.JointPos("q_R_Knee_y", xml_name="R_Knee_y"),
                            ObservationType.JointPos("q_R_Knee_z", xml_name="R_Knee_z"),
                            ObservationType.JointPos("q_R_Ankle_x", xml_name="R_Ankle_x"),
                            ObservationType.JointPos("q_R_Ankle_y", xml_name="R_Ankle_y"),
                            ObservationType.JointPos("q_R_Ankle_z", xml_name="R_Ankle_z"),
                            ObservationType.JointPos("q_R_Toe_x", xml_name="R_Toe_x"),
                            ObservationType.JointPos("q_R_Toe_y", xml_name="R_Toe_y"),
                            ObservationType.JointPos("q_R_Toe_z", xml_name="R_Toe_z"),
                            ObservationType.JointPos("q_Torso_x", xml_name="Torso_x"),
                            ObservationType.JointPos("q_Torso_y", xml_name="Torso_y"),
                            ObservationType.JointPos("q_Torso_z", xml_name="Torso_z"),
                            ObservationType.JointPos("q_Spine_x", xml_name="Spine_x"),
                            ObservationType.JointPos("q_Spine_y", xml_name="Spine_y"),
                            ObservationType.JointPos("q_Spine_z", xml_name="Spine_z"),
                            ObservationType.JointPos("q_Chest_x", xml_name="Chest_x"),
                            ObservationType.JointPos("q_Chest_y", xml_name="Chest_y"),
                            ObservationType.JointPos("q_Chest_z", xml_name="Chest_z"),
                            ObservationType.JointPos("q_Neck_x", xml_name="Neck_x"),
                            ObservationType.JointPos("q_Neck_y", xml_name="Neck_y"),
                            ObservationType.JointPos("q_Neck_z", xml_name="Neck_z"),
                            ObservationType.JointPos("q_Head_x", xml_name="Head_x"),
                            ObservationType.JointPos("q_Head_y", xml_name="Head_y"),
                            ObservationType.JointPos("q_Head_z", xml_name="Head_z"),
                            ObservationType.JointPos("q_L_Thorax_x", xml_name="L_Thorax_x"),
                            ObservationType.JointPos("q_L_Thorax_y", xml_name="L_Thorax_y"),
                            ObservationType.JointPos("q_L_Thorax_z", xml_name="L_Thorax_z"),
                            ObservationType.JointPos("q_L_Shoulder_x", xml_name="L_Shoulder_x"),
                            ObservationType.JointPos("q_L_Shoulder_y", xml_name="L_Shoulder_y"),
                            ObservationType.JointPos("q_L_Shoulder_z", xml_name="L_Shoulder_z"),
                            ObservationType.JointPos("q_L_Elbow_x", xml_name="L_Elbow_x"),
                            ObservationType.JointPos("q_L_Elbow_y", xml_name="L_Elbow_y"),
                            ObservationType.JointPos("q_L_Elbow_z", xml_name="L_Elbow_z"),
                            ObservationType.JointPos("q_L_Wrist_x", xml_name="L_Wrist_x"),
                            ObservationType.JointPos("q_L_Wrist_y", xml_name="L_Wrist_y"),
                            ObservationType.JointPos("q_L_Wrist_z", xml_name="L_Wrist_z"),
                            ObservationType.JointPos("q_L_Hand_x", xml_name="L_Hand_x"),
                            ObservationType.JointPos("q_L_Hand_y", xml_name="L_Hand_y"),
                            ObservationType.JointPos("q_L_Hand_z", xml_name="L_Hand_z"),
                            ObservationType.JointPos("q_R_Thorax_x", xml_name="R_Thorax_x"),
                            ObservationType.JointPos("q_R_Thorax_y", xml_name="R_Thorax_y"),
                            ObservationType.JointPos("q_R_Thorax_z", xml_name="R_Thorax_z"),
                            ObservationType.JointPos("q_R_Shoulder_x", xml_name="R_Shoulder_x"),
                            ObservationType.JointPos("q_R_Shoulder_y", xml_name="R_Shoulder_y"),
                            ObservationType.JointPos("q_R_Shoulder_z", xml_name="R_Shoulder_z"),
                            ObservationType.JointPos("q_R_Elbow_x", xml_name="R_Elbow_x"),
                            ObservationType.JointPos("q_R_Elbow_y", xml_name="R_Elbow_y"),
                            ObservationType.JointPos("q_R_Elbow_z", xml_name="R_Elbow_z"),
                            ObservationType.JointPos("q_R_Wrist_x", xml_name="R_Wrist_x"),
                            ObservationType.JointPos("q_R_Wrist_y", xml_name="R_Wrist_y"),
                            ObservationType.JointPos("q_R_Wrist_z", xml_name="R_Wrist_z"),
                            ObservationType.JointPos("q_R_Hand_x", xml_name="R_Hand_x"),
                            ObservationType.JointPos("q_R_Hand_y", xml_name="R_Hand_y"),
                            ObservationType.JointPos("q_R_Hand_z", xml_name="R_Hand_z"),

                            # ------------- JOINT VEL -------------
                            ObservationType.FreeJointVel("dq_root", xml_name="root"),
                            ObservationType.JointVel("dq_L_Hip_x", xml_name="L_Hip_x"),
                            ObservationType.JointVel("dq_L_Hip_y", xml_name="L_Hip_y"),
                            ObservationType.JointVel("dq_L_Hip_z", xml_name="L_Hip_z"),
                            ObservationType.JointVel("dq_L_Knee_x", xml_name="L_Knee_x"),
                            ObservationType.JointVel("dq_L_Knee_y", xml_name="L_Knee_y"),
                            ObservationType.JointVel("dq_L_Knee_z", xml_name="L_Knee_z"),
                            ObservationType.JointVel("dq_L_Ankle_x", xml_name="L_Ankle_x"),
                            ObservationType.JointVel("dq_L_Ankle_y", xml_name="L_Ankle_y"),
                            ObservationType.JointVel("dq_L_Ankle_z", xml_name="L_Ankle_z"),
                            ObservationType.JointVel("dq_L_Toe_x", xml_name="L_Toe_x"),
                            ObservationType.JointVel("dq_L_Toe_y", xml_name="L_Toe_y"),
                            ObservationType.JointVel("dq_L_Toe_z", xml_name="L_Toe_z"),
                            ObservationType.JointVel("dq_R_Hip_x", xml_name="R_Hip_x"),
                            ObservationType.JointVel("dq_R_Hip_y", xml_name="R_Hip_y"),
                            ObservationType.JointVel("dq_R_Hip_z", xml_name="R_Hip_z"),
                            ObservationType.JointVel("dq_R_Knee_x", xml_name="R_Knee_x"),
                            ObservationType.JointVel("dq_R_Knee_y", xml_name="R_Knee_y"),
                            ObservationType.JointVel("dq_R_Knee_z", xml_name="R_Knee_z"),
                            ObservationType.JointVel("dq_R_Ankle_x", xml_name="R_Ankle_x"),
                            ObservationType.JointVel("dq_R_Ankle_y", xml_name="R_Ankle_y"),
                            ObservationType.JointVel("dq_R_Ankle_z", xml_name="R_Ankle_z"),
                            ObservationType.JointVel("dq_R_Toe_x", xml_name="R_Toe_x"),
                            ObservationType.JointVel("dq_R_Toe_y", xml_name="R_Toe_y"),
                            ObservationType.JointVel("dq_R_Toe_z", xml_name="R_Toe_z"),
                            ObservationType.JointVel("dq_Torso_x", xml_name="Torso_x"),
                            ObservationType.JointVel("dq_Torso_y", xml_name="Torso_y"),
                            ObservationType.JointVel("dq_Torso_z", xml_name="Torso_z"),
                            ObservationType.JointVel("dq_Spine_x", xml_name="Spine_x"),
                            ObservationType.JointVel("dq_Spine_y", xml_name="Spine_y"),
                            ObservationType.JointVel("dq_Spine_z", xml_name="Spine_z"),
                            ObservationType.JointVel("dq_Chest_x", xml_name="Chest_x"),
                            ObservationType.JointVel("dq_Chest_y", xml_name="Chest_y"),
                            ObservationType.JointVel("dq_Chest_z", xml_name="Chest_z"),
                            ObservationType.JointVel("dq_Neck_x", xml_name="Neck_x"),
                            ObservationType.JointVel("dq_Neck_y", xml_name="Neck_y"),
                            ObservationType.JointVel("dq_Neck_z", xml_name="Neck_z"),
                            ObservationType.JointVel("dq_Head_x", xml_name="Head_x"),
                            ObservationType.JointVel("dq_Head_y", xml_name="Head_y"),
                            ObservationType.JointVel("dq_Head_z", xml_name="Head_z"),
                            ObservationType.JointVel("dq_L_Thorax_x", xml_name="L_Thorax_x"),
                            ObservationType.JointVel("dq_L_Thorax_y", xml_name="L_Thorax_y"),
                            ObservationType.JointVel("dq_L_Thorax_z", xml_name="L_Thorax_z"),
                            ObservationType.JointVel("dq_L_Shoulder_x", xml_name="L_Shoulder_x"),
                            ObservationType.JointVel("dq_L_Shoulder_y", xml_name="L_Shoulder_y"),
                            ObservationType.JointVel("dq_L_Shoulder_z", xml_name="L_Shoulder_z"),
                            ObservationType.JointVel("dq_L_Elbow_x", xml_name="L_Elbow_x"),
                            ObservationType.JointVel("dq_L_Elbow_y", xml_name="L_Elbow_y"),
                            ObservationType.JointVel("dq_L_Elbow_z", xml_name="L_Elbow_z"),
                            ObservationType.JointVel("dq_L_Wrist_x", xml_name="L_Wrist_x"),
                            ObservationType.JointVel("dq_L_Wrist_y", xml_name="L_Wrist_y"),
                            ObservationType.JointVel("dq_L_Wrist_z", xml_name="L_Wrist_z"),
                            ObservationType.JointVel("dq_L_Hand_x", xml_name="L_Hand_x"),
                            ObservationType.JointVel("dq_L_Hand_y", xml_name="L_Hand_y"),
                            ObservationType.JointVel("dq_L_Hand_z", xml_name="L_Hand_z"),
                            ObservationType.JointVel("dq_R_Thorax_x", xml_name="R_Thorax_x"),
                            ObservationType.JointVel("dq_R_Thorax_y", xml_name="R_Thorax_y"),
                            ObservationType.JointVel("dq_R_Thorax_z", xml_name="R_Thorax_z"),
                            ObservationType.JointVel("dq_R_Shoulder_x", xml_name="R_Shoulder_x"),
                            ObservationType.JointVel("dq_R_Shoulder_y", xml_name="R_Shoulder_y"),
                            ObservationType.JointVel("dq_R_Shoulder_z", xml_name="R_Shoulder_z"),
                            ObservationType.JointVel("dq_R_Elbow_x", xml_name="R_Elbow_x"),
                            ObservationType.JointVel("dq_R_Elbow_y", xml_name="R_Elbow_y"),
                            ObservationType.JointVel("dq_R_Elbow_z", xml_name="R_Elbow_z"),
                            ObservationType.JointVel("dq_R_Wrist_x", xml_name="R_Wrist_x"),
                            ObservationType.JointVel("dq_R_Wrist_y", xml_name="R_Wrist_y"),
                            ObservationType.JointVel("dq_R_Wrist_z", xml_name="R_Wrist_z"),
                            ObservationType.JointVel("dq_R_Hand_x", xml_name="R_Hand_x"),
                            ObservationType.JointVel("dq_R_Hand_y", xml_name="R_Hand_y"),
                            ObservationType.JointVel("dq_R_Hand_z", xml_name="R_Hand_z")]

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
            # Left leg
            "L_Hip_x", "L_Hip_y", "L_Hip_z",
            "L_Knee_x", "L_Knee_y", "L_Knee_z",
            "L_Ankle_x", "L_Ankle_y", "L_Ankle_z",
            "L_Toe_x", "L_Toe_y", "L_Toe_z",
            # Right leg
            "R_Hip_x", "R_Hip_y", "R_Hip_z",
            "R_Knee_x", "R_Knee_y", "R_Knee_z",
            "R_Ankle_x", "R_Ankle_y", "R_Ankle_z",
            "R_Toe_x", "R_Toe_y", "R_Toe_z",
            # Torso and spine
            "Torso_x", "Torso_y", "Torso_z",
            "Spine_x", "Spine_y", "Spine_z",
            "Chest_x", "Chest_y", "Chest_z",
            "Neck_x", "Neck_y", "Neck_z",
            "Head_x", "Head_y", "Head_z",
            # Left arm
            "L_Thorax_x", "L_Thorax_y", "L_Thorax_z",
            "L_Shoulder_x", "L_Shoulder_y", "L_Shoulder_z",
            "L_Elbow_x", "L_Elbow_y", "L_Elbow_z",
            "L_Wrist_x", "L_Wrist_y", "L_Wrist_z",
            "L_Hand_x", "L_Hand_y", "L_Hand_z",
            # Right arm
            "R_Thorax_x", "R_Thorax_y", "R_Thorax_z",
            "R_Shoulder_x", "R_Shoulder_y", "R_Shoulder_z",
            "R_Elbow_x", "R_Elbow_y", "R_Elbow_z",
            "R_Wrist_x", "R_Wrist_y", "R_Wrist_z",
            "R_Hand_x", "R_Hand_y", "R_Hand_z"
        ]

        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the Unitree H1 environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "SMPLBoxing" / "smpl_humanoid_neutral_boxing.xml").as_posix()

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body specified in the XML file.
        """
        return "Torso"

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
        return (0.6, 1.5)
