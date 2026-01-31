# Autonomous_Driving_Hackathon
경북대학교 AICOSS 자율주행로봇 경진대회

Autonomous Racing: RL & CV Integration

Team 애드 아스트라

1. Event Overview
This project was developed for a high-intensity hackathon focused on navigating a "Digital 8" track using the AutoCar III G platform. The challenge required the vehicle to autonomously identify complex intersections and alternate between left and right loops while maintaining stability on both high-friction carpet and smooth track surfaces.

2. Reinforcement Learning (RL) Framework
The core intelligence for our decision-making logic was built using a Grand Champion Ensemble of specialized Multi-Armed Bandit (MAB) agents. This architecture was designed to handle abrupt-shift environments, where reward probabilities change suddenly.

Algorithm Architecture
The framework utilizes a Council of Experts approach to ensure high adaptability across diverse data distributions:

MirrorCusumExpert Council: We deployed an ensemble of four distinct agents, each tuned with specific thresholds and drift parameters to specialize in different data "personalities".

Double-Mirror Posterior Update: The agents employ a unique update rule where a reward for one arm simultaneously updates the posterior of the non-selected arm (Mirroring), effectively doubling the learning speed.

Double-CUSUM Detection: Each expert features a built-in change-point detection mechanism (g+ and g−) that monitors for statistical drift. If a shift in reward probability exceeds the defined threshold, the expert triggers a local reset to re-learn the new environment.

Virtual Accuracy History: The top-level BanditAgent tracks a sliding window of "virtual rewards" for each council member, dynamically routing control to the expert currently showing the highest historical accuracy.

<img width="264" height="235" alt="AL_result" src="https://github.com/user-attachments/assets/04f4540a-1d44-41d9-afb6-d5cb73abe0a9" />

As a result, our RL model scored a total score of 1345, which was 2nd out of 10 teams.
You can see the details in the docs folder.

3. Autonomous Driving Model (Residual CNN)
To execute the turns, we developed a Deep Learning regression model designed to run on the NVIDIA brain board.

Architecture: A 9-layer Residual CNN with Swish activation and Batch Normalization for smooth steering prediction.

Preprocessing: We implemented 2500 images and labeled each image, ensuring the model could treat diagonal curves as vertical paths for better centering.
Due to time limitation, we focused on programming the car to turn left on the digital 8 track.

![228_228_2026-01-29_17-00-01](https://github.com/user-attachments/assets/dbb47c12-2d2a-4cec-ba2c-f1b72215d289)
![286_286_2026-01-29_16-17-30](https://github.com/user-attachments/assets/31b118b9-97e5-46e7-9fe3-dbf8cecd5e19)
![66_66_2026-01-29_16-45-24](https://github.com/user-attachments/assets/8a42b803-1acc-4817-acc9-cc364f70c088)


Adaptive Control: The system uses the DC motor with encoder to dynamically scale speed—slowing down by 60% during Status 3 (Turns) and accelerating on straights.




