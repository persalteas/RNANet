# This is a script supposed to be run periodically as a cron job

cd /home/lbecquey/Projects/RNANet
rm -rf latest_run.log errors.txt

# Run RNANet
bash -c 'time python3.8 ./RNAnet.py --3d-folder /home/lbecquey/Data/RNA/3D/ --seq-folder /home/lbecquey/Data/RNA/sequences/ -r 20.0 --redundant --sina --extract -s --stats-opts="--wadley --distance-matrices" --archive' > latest_run.log 2>&1
echo 'Compressing RNANet.db.gz...' >> latest_run.log
touch results/RNANet.db                                         # update last modification date
gzip -k /home/lbecquey/Projects/RNANet/results/RNANet.db        # compress it
rm -f results/RNANet.db-wal results/RNANet.db-shm               # SQLite temporary files

# Save the latest results
export DATE=`date +%Y%m%d`
echo "Creating new release in ./archive/ folder ($DATE)..." >> latest_run.log
cp /home/lbecquey/Projects/RNANet/results/summary.csv /home/lbecquey/Projects/RNANet/archive/summary_latest.csv
cp /home/lbecquey/Projects/RNANet/results/summary.csv "/home/lbecquey/Projects/RNANet/archive/summary_$DATE.csv"
cp /home/lbecquey/Projects/RNANet/results/families.csv /home/lbecquey/Projects/RNANet/archive/families_latest.csv
cp /home/lbecquey/Projects/RNANet/results/families.csv "/home/lbecquey/Projects/RNANet/archive/families_$DATE.csv"
cp /home/lbecquey/Projects/RNANet/results/frequencies.csv /home/lbecquey/Projects/RNANet/archive/frequencies_latest.csv
cp /home/lbecquey/Projects/RNANet/results/pair_types.csv /home/lbecquey/Projects/RNANet/archive/pair_types_latest.csv
mv /home/lbecquey/Projects/RNANet/results/RNANet.db.gz /home/lbecquey/Projects/RNANet/archive/

# Init Seafile synchronization between RNANet library and ./archive/ folder (just the first time !)
# seaf-cli sync -l 8e082c6e-b9ed-4b2f-9279-de2177134c57 -s https://entrepot.ibisc.univ-evry.fr -u l****.b*****y@univ-evry.fr -p ****************** -d archive/

# Sync in Seafile
seaf-cli start >> latest_run.log 2>&1
echo 'Waiting 10m for SeaFile synchronization...' >> latest_run.log
sleep 10m
echo `seaf-cli status` >> latest_run.log
seaf-cli stop >> latest_run.log 2>&1
echo 'We are '`date`', update completed.' >> latest_run.log

