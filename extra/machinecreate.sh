sudo apt update && sudo apt upgrade -y
sudo apt install python3-venv python3-pip -y
python3 -m venv myenv
source myenv/bin/activate
pip install google-api-python-client google-auth-oauthlib 
screen -ls -r


pip install -r requirements.txt
nano requirements.txt

python3 main.py download 1fu_1CCeKw42zl-iCjEcQodA5reFlFLn2


nano active.txt
source myenv/bin/activate
