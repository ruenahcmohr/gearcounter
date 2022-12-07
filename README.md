# gearcounter
opencv software to count teeth on gears - python


This version uses two algorythms to count teeth:
1) take the perimeter trace of the gear, convert it to polar, offset the resulting waveform by its average and count the zero crossings.

2) take a ring 10 pixels in from the edge of the circle and create an intensity waveform from the greyscale version of image, run this waveform thru an FFT.

If everything works nicely there will be two counts for each gear, and ideally, they will agree.
The FFT is drawn last, so its count will match the circle around the perimeter of the gear. Its more-often-than-not correct.


## MacOS installation

```shell
# create new miniconda environment:
conda create -p $PWD/.venv
# activate it 
conda activate $PWD/.venv
pip install -r requirements.txt
```

Run via:

```shell
python3 gearcounter3.py srcfile.jpg
```



-----

known failures, not understood yet:
(but the counter does NOT like lint!!!)

18 
38
41
59
65
67
78 WHY???



hopless images:
76
06
