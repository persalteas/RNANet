# This is a script supposed to be run periodically as a cron job

cd /home/lbecquey/Projects/RNANet
rm -f latest_run.log errors.txt

# Run RNANet
bash -c 'time ./RNAnet.py --3d-folder /home/lbecquey/Data/RNA/3D/ --seq-folder /home/lbecquey/Data/RNA/sequences/ -r 20.0 -s --archive' &> latest_run.log
touch results/RNANet.db # update last modification date
rm -f results/RNANet.db-wal results/RNANet.db-shm # SQLite temporary files

# Compress
rm -f /home/lbecquey/Projects/RNANet/results/RNANet.db.gz
echo 'Deleted results/RNANet.db.gz (if existed)' >> latest_run.log
gzip -k /home/lbecquey/Projects/RNANet/results/RNANet.db
echo 'Recreated it.' >> latest_run.log

# Sync in Seafile
seaf-cli start >> latest_run.log 2>&1
echo 'Waiting 10m for SeaFile synchronization...' >> latest_run.log
sleep 10m
echo `seaf-cli status` >> latest_run.log
seaf-cli stop >> latest_run.log 2>&1
echo 'We are '`date`', update completed.' >> latest_run.log

