# PAIRS: Selective Face Protection through Aggregated Input-to-Representation Sensitivity
## Environment Setup
```bash
conda create -n face-protection python=3.10 -y
conda activate face-protection
pip install -r requirements.txt
```
## Run Protect a face image:
```bash
python demo.py --image /path/to/face.jpg
```

To evaluate face verification, provide another image of the same identity:
```bash
python demo.py --image /path/to/face_1.jpg --reference /path/to/face_2.jpg
```
The protected image is saved in outputs/. Similarity scores and MATCH / NO MATCH results are printed in the terminal.
