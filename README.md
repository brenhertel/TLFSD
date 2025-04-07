# TLFSD: Trajectory Learning via Failed and Successful Demonstrations

Implementation of Trajectory Learning via Failed and Successful Demonstrations (TLFSD)

Corresponding paper can be found for free [here](https://arxiv.org/abs/2107.11918), please read for method details.

Learning from Demonstration (LfD) is a popular approach that allows humans to teach robots new skills by showing the correct way(s) of performing the desired skill. Human-provided demonstrations, however, are not always optimal and the teacher usually addresses this issue by discarding or replacing sub-optimal (noisy or faulty) demonstrations. We propose a novel LfD representation that learns from both successful and failed demonstrations of a skill. Our approach encodes the two subsets of captured demonstrations (labeled by the teacher) into a statistical skill model, constructs a set of quadratic costs, and finds an optimal reproduction of the skill under novel problem conditions (i.e. constraints). The optimal reproduction balances convergence towards successful examples and divergence from failed examples. We evaluate our approach through several 2D and 3D experiments in real-world using a UR5e manipulator arm and also show that it can reproduce a skill from only failed demonstrations. The benefits of exploiting both failed and successful demonstrations are shown through comparison with two existing LfD approaches. We also compare our approach against an existing skill refinement method and show its capabilities in a multi-coordinate setting.

<img src="https://github.com/brenhertel/TLFSD/blob/main/pictures/paper_figures/reaching_2D.png" alt="" width="318"/> <img src="https://github.com/brenhertel/TLFSD/blob/main/pictures/paper_figures/robot_reaching.png" alt="" width="300"/>

This repository implements the method described in the paper above using Python. Scripts which perform individual experiments are included, as well as other necessary utilities. If you have any questions, please contact Brendan Hertel (brendan_hertel@student.uml.edu).

If you use the code present in this repository, please cite the following paper:
```
@inproceedings{hertel2021TLFSD,
  title={Learning from Successful and Failed Demonstrations via Optimization},
  author={Brendan Hertel and S. Reza Ahmadzadeh},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={7807--7812},
  year={2021},
  organization={IEEE}
}
```

### Non-TLFSD algorithms implemented here:

Dynamic Movement Primitives (DMPs) as presented by P. Pastor, H. Hoffmann, T. Asfour, and S. Schaal, “Learning and generalization of motor skills by learning from demonstration,” in IEEE International Conference on Robotics and Automation. IEEE, 2009, pp. 763–768. Implementation can be found [here](https://github.com/carlos22/pydmp)

Laplacian Trajectory Editing (LTE) as presented by T. Nierhoff, S. Hirche, and Y. Nakamura, “Spatial adaption of robot trajectories based on laplacian trajectory editing,” Autonomous Robots, vol. 40, no. 1, pp. 159–173, 2016. Implementation based on MATLAB code provided by the authors which is no longer publicly available.

Gaussian Mixture Models/Gaussian Mixture Regression (GMM/GMR) as presented by S. Calinon, F. Guenter, and A. Billard, “On learning, representing, and generalizing a task in a humanoid robot,” IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 37, no. 2,pp. 286–298, 2007. Implementation can be found [here](https://github.com/BatyaGG/Gaussian-Mixture-Models)

Gaussian Mixture Models/Gaussian Mixture Regression with weighted Expectation-Maximization (GMM/GMR-wEM) as presented by B. D. Argall, E. L. Sauser, and A. G. Billard, “Tactile guidance for policy refinement and reuse,”  in IEEE 9th International Conference on Development and Learning. IEEE, 2010, pp. 7–12. Original implementation based on paper.
