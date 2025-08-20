sudo apt update && sudo apt upgrade -y
sudo apt install python3-venv python3-pip -y
python3 -m venv myenv
source myenv/bin/activate
pip install google-api-python-client google-auth-oauthlib 
echo screen -ls -r