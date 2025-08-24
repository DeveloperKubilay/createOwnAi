lsblk
sudo mkswap /dev/sdb
sudo swapon /dev/sdb

sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile


sudo swapoff /swapfile
sudo rm /swapfile
