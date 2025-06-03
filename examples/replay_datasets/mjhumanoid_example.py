import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment(seed=0):

    np.random.seed(seed)

    # # example --> you can add as many datasets as you want in the lists!
    env = ImitationFactory.make("MJHumanoid",
                                # if SMPL and AMASS are installed, you can use the following:
                                # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"]),
                                amass_dataset_conf=AMASSDatasetConf(["CMU/CMU/13/venia_dancing"]),
                                # lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
                                n_substeps=15)

    env.play_trajectory(n_episodes=3, n_steps_per_episode=None, render=True)


if __name__ == '__main__':
    experiment()
