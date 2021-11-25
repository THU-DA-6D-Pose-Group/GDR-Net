# some other dependencies
set -x
sudo apt-get install libjpeg-dev zlib1g-dev
sudo apt-get install libopenexr-dev
sudo apt-get install openexr
sudo apt-get install python3-dev
sudo apt-get install libglfw3-dev libglfw3
sudo apt-get install libassimp-dev

# conda install ipython
pip install -r requirements.txt

pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
