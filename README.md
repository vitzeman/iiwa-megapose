# iiwa-megapose
Repository for my DT with megapsose running on iiwa
Věta


# RUNNING(Intended)
Firs prepare the scrip at cluster
```
conda activate megapose
cd ~/Projects/megapose6d
python Server.py

```
Then run the script at the robot
Prepare the robot 
<!-- itemize -->
- Turn on the robot
- On tablet run  the script TODO: ADD NAME
- Check camera connection in Pylon Viewer
- Check the IP address of the robot
- Check the IP address of the cluster

Run the python script after the cluster script is running
```
conda activate megapose
cd ~/Projects/iiwa-megapose
python pick_main.py
```