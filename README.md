# OpticNerve
OpticNerve is a real-time multi-face recognition system, and provide friendly web interface to use the system.

## Installation
### MongoDB
- [MongoDB (Enterprise)](https://www.mongodb.com/download-center/enterprise)

### Python
- [python 3.6.8](https://www.python.org/downloads/)

### Package
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install reqirments.txt.

```bash
 pip install -r requirements.txt
```

### CUDA (optional)
You can choose installation cuda or not, face recognition system can use gpu cuda accelerate.
- For GTX-1050ti gpu
    - cuda 9.2 [here](https://developer.nvidia.com/cuda-92-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)
    - cuDNN v7.6.5 [here](https://developer.nvidia.com/rdp/cudnn-download)
    - Installation step [here](https://hackmd.io/@conflick0/OpticNerve#CUDA)

## Usage
1. Import face_db to MongoDB database.
```bash
mongorestore -d face_db --dir ./face_db
```
2. Run main.py.
```bash
python main.py
```
3. Open the browser, and input url http://127.0.0.1:5000/.

![](https://i.imgur.com/3Z22F68.png)

4. Strat to use the system by web inteface.

![](https://i.imgur.com/F1gQ705.png)

---
## Feature
    
### Recognize
1. Click **Recognize** in the home page navigation.

![](https://i.imgur.com/kSGGmYB.jpg)

2. Start recognize.

![](https://i.imgur.com/sj2Ax97.png)

### Sign up
1. Click **Sign up** button in the home page.

![](https://i.imgur.com/bFgeksf.jpg)

2. Input name.

![](https://i.imgur.com/98VUdLq.png)

3. Choose personal infomation.

![](https://i.imgur.com/giZ3xSZ.png)

4. Click **Capture** button to take a picture.

![](https://i.imgur.com/CeeepZV.png)

5. Click **Finish** button to finish .

![](https://i.imgur.com/FGr4tZy.png)

6. Finish sign up.

![](https://i.imgur.com/OWnRAcF.png)

### Management
- Check user information, remove user and reset(clear) database.

![](https://i.imgur.com/KNc7t07.png)

![](https://i.imgur.com/DnpeSbo.png)

![](https://i.imgur.com/k1cPwe3.png)

### Log
- Check user recognized time and information.

![](https://i.imgur.com/AlfzlAr.png)

![](https://i.imgur.com/jwusPIX.png)
