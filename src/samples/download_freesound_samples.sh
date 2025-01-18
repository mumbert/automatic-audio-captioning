#!/bin/bash

# FOLDER NAME
FREESOUND_FOLDER="$HOME/automatic-audio-captioning/samples/freesound/"

# CLEAN AND CREATE FOLDER
[ -d $FREESOUND_FOLDER ] && rm -fr $FREESOUND_FOLDER
mkdir -p $FREESOUND_FOLDER
cd $FREESOUND_FOLDER

# DOWNLOAD FILES
wget https://freesound.org/people/klankbeeld/sounds/784190/download/784190__klankbeeld__village-bokhoven-nl-0944-am-150405_0586.wav

