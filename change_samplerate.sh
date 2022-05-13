#!/bin/bash

ORIG=$1 #directory that contain original audio
SOX=$2 #direcory where converted audio will be placed

subdirs=$(ls $ORIG);
cd $ORIG
for dir in $subdirs; do
    mkdir "../$SOX/$dir"
    for audio in $(ls $dir); do
        sox -v 0.86 "$dir/$audio"\
        --bits 16 --no-dither --compression 0.0\
        "../$SOX/$dir/$audio"\
        channels 1 rate 16000 || echo $audio corrupted
    done
done