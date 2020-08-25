# This is a script supposed to be run periodically as a cron job

cd /home/lbecquey/Projects/RNANet;
rm -f nohup.out errors.txt;

# Run RNANet
nohup bash -c 'time ./RNAnet.py --3d-folder /home/lbecquey/Data/RNA/3D/ --seq-folder /home/lbecquey/Data/RNA/sequences/ -s -r 20.0 --archive';

# Compress
rm -f results/RNANet.db.gz
gzip -k results/RNANet.db

# Sync in Seafile
seaf-cli start;
sleep 30m;
seaf-cli stop;

