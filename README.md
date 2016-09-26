# Photo Auto Balancer

A quick fun project which utilises Deep Learning to 
automate image color/level adjustment.

---

## Quick Start

Install all prerequisites with the following command.

```bash
$ pip install -r requirements.txt
```

### Prepare training directory

Say your preferred trainset directory is `/path/to/trainset/`. 
Create three subdirectories inside it as follows.

```
trainset
├── out/
├── raw/
└── unfiltered/
```

Then put your images as a trainset inside `/path/to/trainset/raw`.
Leave the other two empty. You're all set.

### Training

```bash
$ python3 loader.py --train --limit 30 --dir /path/to/trainset/
```

The script reads all JPG images from the `dir` you specified 
in the arguments. The reverse filtered images will be generated inside 
`out` subdirectory. 

> *CAVEAT*: The process starts training Convolutional Neural Network 
rightaway after the reverse filtered samples are generated. 
This takes **huge computational power and time**.

---

## Licence

MIT licence.