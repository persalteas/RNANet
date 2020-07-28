# This is a script supposed to be run periodically as a cron job

# Run RNANet
cd /home/lbecquey/Projects/RNANet;
rm -f stdout.txt stderr.txt errors.txt;
time './RNAnet.py --3d-folder /home/lbequey/Data/RNA/3D/ --seq-folder /home/lbecquey/Data/RNA/sequences/ -s -r 20.0' > stdout.txt 2> stderr.txt;

# Sync in Seafile
seaf-cli start;

seaf-cli stop;

