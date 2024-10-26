# Priv-FedTL
Code base for our implementation of Priv-FedTL. Follow the instructions below to setup the server and client devices for conducting the experiments.

## Environment setup
On both types of devices execute the following commands to create and activate a virtual environment named 'testenv' (you can change this name to your preference):
```
python3 -m venv testenv 
```
```
source activate testenv/bin/activate
```
**Server - Linux (Debian)** :
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```
pip3 install h5py
```
```
pip3 install -U scikit-learn scipy matplotlib
```
```
pip3 install tqdm
```
```
pip install tenseal
```
**Client - RaspbianOS** :
We used the Raspberry Pi 4 Model B as the client device. Follow the steps below to setup the Rasberry Pi:
1. Download [custom OS](https://github.com/Qengineering/RPi-Bullseye-DNN-image) image file and burn it in a 32 GB sd card
2. Uninstall Paddle-Lite if not needed as it takes a lot of storage:
```	
du -s * | sort -nr | head -n10
sudo rm -rf ./Paddle-Lite && rm -rf ./PaddleOCR && rm -rf ./MNN
```	
3. Fix the pytorch lib issue:
```
cd /usr/local/lib/python3.9/dist-packages/torch/
sudo chmod 777 ./storage.py
nano ./storage.py
```
Add this in the storage.py file
```
	if torch.cuda.is_available():
		return torch.load(io.BytesIO(b))
	else:
		return torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
```
4. Build TenSEAL for Raspberry Pi:
```	
git clone https://github.com/OpenMined/TenSEAL.git
pip install cmake
```
```
nano ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```
```
cd TenSEAL
sudo python setup.py install
```
5. Clone this git repo

6. Install other dependencies:
```	
pip3 install -U scikit-learn scipy matplotlib
pip install typing-extensions==4.3.0
pip3 install tqdm
```
