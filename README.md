# iiwa-megapose
Repository for my DT with megapsose running on iiwa

# RUNNING(Intended)
## 1. **cluster**
After connecting to the cluster, run the following commands
```
conda activate megapose
cd ~/Projects/megapose6d
python Server.py
```

## 2. **robot KMR**
Prepare the robot 
<!-- itemize -->
- Turn on the robot
- On tablet run  the script OpcUaMoveTo

## 3. **Check the conections**
Make sure that the following connections are working
- Check camera connection in Pylon Viewer
- Check the IP address of the robot
- Check the IP address of the cluster

## 4. **Run the PC  script**
Run the python script after the cluster script is running
```
conda activate megapose
cd ~/Projects/iiwa-megapose
python pick_main.py
```