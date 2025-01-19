#!/bin/bash

# FOLDER NAME
SAMPLE_FOLDER="$HOME/automatic-audio-captioning/samples/wikipedia/"

# CLEAN AND CREATE FOLDER
[ -d $SAMPLE_FOLDER ] && rm -fr $SAMPLE_FOLDER
mkdir -p $SAMPLE_FOLDER
cd $SAMPLE_FOLDER

# DOWNLOAD FILES
wget https://upload.wikimedia.org/wikipedia/commons/4/40/Toreador_song_cleaned.ogg

