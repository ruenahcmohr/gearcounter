# gearcounter
opencv software to count teeth on gears - python

The input filename is fixed to foo.jpg - sorry, its just too hard to change that part of the code.

This version uses two algorythms to count teeth:
1) take the perimeter trace of the gear, convert it to polar, offset the resulting waveform by its average and count the zero crossings.

2) take a ring 10 pixels in from the edge of the circle and create an intensity waveform from the greyscale version of image, run this waveform thru an FFT.

If everything works nicely there will be two counts for each gear, and ideally, they will agree.
The FFT is drawn last, so its count will match the circle around the perimeter of the gear. Its more-often-than-not correct.


