sudo apt install python3-venv python3-pip -y
python3 -m venv myenv
source myenv/bin/activate
pip install google-api-python-client google-auth-oauthlib
echo source myenv/bin/activate > active.txt
screen

pip install -r requirements.txt
nano requirements.txt
source myenv/bin/activate

python3 main.py download 1fu_1CCeKw42zl-iCjEcQodA5reFlFLn2
mv 1fu_1CCeKw42zl-iCjEcQodA5reFlFLn2 ./model.jsonl
python3 main.py download 1sdApIUSmXpAtlA5f1CAOORxPbfoGo2qk
mv 1sdApIUSmXpAtlA5f1CAOORxPbfoGo2qk ../model_tokenized.jsonl

echo "PS1='\[\e[38;5;71m\]\u@\h\[\e[38;5;15m\]:\[\e[38;5;19m\]\w \[\e[38;5;15m\]#\[\e[0m\] '" >> ~/.bashrc && source ~/.bashrc
sudo apt install unzip htop -y
sudo nano /etc/ssh/sshd_config
PasswordAuthentication yes
PermitRootLogin yes
sudo passwd 
sudo systemctl restart sshd

pip install torch transformers datasets accelerate
pip install tokenizers orjson transformers TOKANİZER İÇİN BU