import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment(seed=0):

    np.random.seed(seed)

    import os
    from glob import glob

    # Get all motion files from the BMLhandball dataset for different subjects
    subjects = ["S01_Expert", "S02_Novice", "S03_Expert", "S04_Expert", "S05_Novice", 
                "S06_Novice", "S07_Expert", "S08_Novice", "S09_Novice", "S10_Expert"]
    
    motion_files = []
    for subject in subjects:
        amass_dir = os.path.join(f"amass/BMLhandball/{subject}")
        subject_files = [f.replace(".npz", "").replace("amass/", "")
                        for f in glob(os.path.join(amass_dir, "*_poses.npz"))]
        motion_files.extend(subject_files)
    
    # # example --> you can add as many datasets as you want in the lists!
    env = ImitationFactory.make("SMPLBoxing",
                                # if SMPL and AMASS are installed, you can use the following:
                                # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"]),
                                amass_dataset_conf=AMASSDatasetConf(["CMU/CMU/13/13_17_poses"]),
                                # amass_dataset_conf=AMASSDatasetConf(motion_files),
                                # lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
                                n_substeps=15)

    env.play_trajectory(n_episodes=3, n_steps_per_episode=None, render=True)


if __name__ == '__main__':
    experiment()
