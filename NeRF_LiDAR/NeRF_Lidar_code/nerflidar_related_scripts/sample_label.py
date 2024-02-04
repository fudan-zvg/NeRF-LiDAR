import os
import numpy as np
scene_list = [1,3,5,11,14,16,23,29,33]
for i in scene_list:
    os.system("rm -rf 6cam_scene{}_revision/sample_labels".format(i))
    os.system("cp -r ./semantickitti_nusc_sample/scene{} 6cam_scene{}_revision/sample_labels".format(i,i))