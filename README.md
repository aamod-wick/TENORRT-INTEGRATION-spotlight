# Unsafe Way 1: 
## STEP 1 : git clone   
git clone https://github.com/aamod-wick/TENORRT-INTEGRATION-spotlight
## STEP 2 : set up enviroment  
if not in directory already  
cd TENORRT-INTEGRATION-spotlight   
chmod +x setup.sh  
./setup.sh  
## STEP 3 : install model onnx files     
install in same folder from https://drive.google.com/drive/folders/1mxF7L3P0KYyaINJQWW8W49Mqs91_mEyl
## STEP 4 : python3 main.py   

# ALTERNATIVE WAY 
## STEP 1 : git clone   
git clone https://github.com/aamod-wick/TENORRT-INTEGRATION-spotlight
## STEP 2 : set up enviroment   
if not in directory already  
cd TENORRT-INTEGRATION-spotlight   
python3 -m venv env  
source env/bin/activate  
pip install requirements.txt  

## STEP 3 : install model onnx files   
install in same folder from https://drive.google.com/drive/folders/1mxF7L3P0KYyaINJQWW8W49Mqs91_mEyl  
## STEP 4 : python3 main.py   

