# This is a script supposed to be run periodically as a cron job

cd /home/lbecquey/Projects/RNANet;
rm -f stdout.txt stderr.txt errors.txt;

# Run RNANet
time './RNAnet.py --3d-folder /home/lbequey/Data/RNA/3D/ --seq-folder /home/lbecquey/Data/RNA/sequences/ -s -r 20.0 --archive' > stdout.txt 2> stderr.txt;

# Compress
rm -f results/RNANet.db.gz
gzip -k results/RNANet.db

# Sync in Seafile
seaf-cli start;
sleep 30m;
seaf-cli stop;

